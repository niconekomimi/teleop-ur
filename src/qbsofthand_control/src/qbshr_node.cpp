#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include <serial/serial.h>

#include <qbrobotics_research_api/qbsofthand_research_api.h>

#include <qbsofthand_control/srv/set_closure.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <optional>
#include <regex>

class QBSoftHandNode : public rclcpp::Node {
public:
    QBSoftHandNode() : Node("qbsofthand_control_node") {
        declare_parameter<std::string>("serial_port", "");
        declare_parameter<int>("device_id", -1);
        declare_parameter<int>("max_repeats", 2);

        declare_parameter<bool>("auto_activate", true);
        declare_parameter<double>("command_rate_hz", 50.0);

        declare_parameter<int>("open_reference", 0);
        declare_parameter<int>("close_reference", 19000);

        // Speed-based mode uses incremental steps per timer tick.
        // Actual step = max_step_per_tick * speed_ratio.
        declare_parameter<int>("max_step_per_tick", 300);

        serial_port_ = get_parameter("serial_port").as_string();
        device_id_ = get_parameter("device_id").as_int();
        max_repeats_ = get_parameter("max_repeats").as_int();
        auto_activate_ = get_parameter("auto_activate").as_bool();
        command_rate_hz_ = get_parameter("command_rate_hz").as_double();
        open_reference_ = static_cast<int16_t>(get_parameter("open_reference").as_int());
        close_reference_ = static_cast<int16_t>(get_parameter("close_reference").as_int());

        max_step_per_tick_ = get_parameter("max_step_per_tick").as_int();
        if (max_step_per_tick_ <= 0) {
            max_step_per_tick_ = 300;
        }

        if (command_rate_hz_ <= 0.0) {
            command_rate_hz_ = 50.0;
        }

        initDevice();

        set_closure_srv_ = create_service<qbsofthand_control::srv::SetClosure>(
            "~/set_closure",
            std::bind(&QBSoftHandNode::onSetClosure, this, std::placeholders::_1, std::placeholders::_2));

        activate_srv_ = create_service<std_srvs::srv::SetBool>(
            "~/activate",
            std::bind(&QBSoftHandNode::onActivate, this, std::placeholders::_1, std::placeholders::_2));

        timer_ = create_wall_timer(
            std::chrono::duration<double>(1.0 / command_rate_hz_),
            std::bind(&QBSoftHandNode::onTimer, this));

        RCLCPP_INFO(get_logger(), "qbsofthand_control_node started");
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        motion_.active = false;
        timer_.reset();
        // Release device objects while executor is no longer spinning.
        // Avoid calling explicit close here to prevent double-free in third-party libs.
        hand_.reset();
        communication_.reset();
    }

private:
    enum class ControlMode { Duration, Speed };

    struct Motion {
        bool active{false};
        ControlMode mode{ControlMode::Duration};
        int16_t start_ref{0};
        int16_t current_ref{0};
        int16_t target_ref{0};
        rclcpp::Time start_time{0, 0, RCL_ROS_TIME};
        rclcpp::Duration duration{0, 0};
        float speed_ratio{1.0f};
    };

    std::mutex mutex_;
    Motion motion_;

    std::shared_ptr<qbrobotics_research_api::CommunicationLegacy> communication_;
    std::shared_ptr<qbrobotics_research_api::qbSoftHandLegacyResearch> hand_;

    std::string serial_port_;
    int device_id_{-1};
    int max_repeats_{2};
    bool auto_activate_{true};
    double command_rate_hz_{50.0};
    int16_t open_reference_{0};
    int16_t close_reference_{19000};
    int max_step_per_tick_{300};

    std::optional<int16_t> last_sent_reference_;

    rclcpp::Service<qbsofthand_control::srv::SetClosure>::SharedPtr set_closure_srv_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr activate_srv_;
    rclcpp::TimerBase::SharedPtr timer_;

    static float clampFloat(float value, float lo, float hi) {
        return std::min(std::max(value, lo), hi);
    }

    static int16_t clampInt16(int value) {
        return static_cast<int16_t>(std::min(std::max(value, -32768), 32767));
    }

    bool openSerialPortWithRetries(const std::string &port_name) {
        int failures = 0;
        while (failures <= max_repeats_) {
            if (communication_->openSerialPort(port_name) >= 0) {
                return true;
            }
            failures++;
        }
        return false;
    }

    void maybeInferCloseReferenceFromDevice() {
        if (!hand_) {
            return;
        }
        uint8_t use_position_limits = 0;
        std::vector<int32_t> position_limits;
        if (hand_->getParamUsePositionLimits(use_position_limits) != 0) {
            return;
        }
        if (use_position_limits == 0) {
            return;
        }
        if (hand_->getParamPositionLimits(position_limits) != 0) {
            return;
        }
        if (position_limits.size() < 2) {
            return;
        }

        // SoftHand 通常用单输入(协同)做闭合；这里优先用第一个上限作为 close_reference。
        const int32_t upper = position_limits.at(1);
        if (upper > 0) {
            close_reference_ = clampInt16(static_cast<int>(upper));
        }
    }

    void initDevice() {
        communication_ = std::make_shared<qbrobotics_research_api::CommunicationLegacy>();

        std::vector<std::string> candidate_ports;
        if (!serial_port_.empty()) {
            candidate_ports.push_back(serial_port_);
        } else {
            std::vector<serial::PortInfo> serial_ports;
            if (communication_->listSerialPorts(serial_ports) < 0) {
                RCLCPP_ERROR(get_logger(), "No serial ports found");
                return;
            }

            // 默认只尝试 /dev/ttyUSB*，避免误打开别的串口设备
            const std::regex port_regex("/dev/ttyUSB[[:digit:]]+");
            for (const auto &port : serial_ports) {
                if (std::regex_match(port.serial_port, port_regex)) {
                    candidate_ports.push_back(port.serial_port);
                }
            }
        }

        std::vector<qbrobotics_research_api::Communication::ConnectedDeviceInfo> device_ids;

        for (const auto &port_name : candidate_ports) {
            if (!openSerialPortWithRetries(port_name)) {
                RCLCPP_WARN(get_logger(), "Failed to open port: %s", port_name.c_str());
                continue;
            }
            RCLCPP_INFO(get_logger(), "Opened port: %s", port_name.c_str());

            device_ids.clear();
            if (communication_->listConnectedDevices(port_name, device_ids) < 0) {
                continue;
            }

            for (const auto &dev : device_ids) {
                if (dev.id == 0 || dev.id == 120) {
                    continue;
                }
                if (device_id_ > 0 && dev.id != static_cast<uint8_t>(device_id_)) {
                    continue;
                }

                hand_ = std::make_shared<qbrobotics_research_api::qbSoftHandLegacyResearch>(
                    communication_, "qbsofthand", port_name, dev.id);
                serial_port_ = port_name;
                device_id_ = static_cast<int>(dev.id);

                RCLCPP_INFO(get_logger(), "Connected to qbSoftHand id=%d on %s", device_id_, serial_port_.c_str());
                maybeInferCloseReferenceFromDevice();

                if (auto_activate_) {
                    (void)hand_->setMotorStates(true);
                }
                return;
            }
        }

        RCLCPP_ERROR(get_logger(), "No qbSoftHand devices found (serial_port='%s', device_id=%d)", serial_port_.c_str(), device_id_);
    }

    bool sendReference(int16_t reference) {
        if (!hand_) {
            return false;
        }
        std::vector<int16_t> refs{reference};
        const int ret = hand_->setControlReferences(refs);
        if (ret == 0) {
            last_sent_reference_ = reference;
            return true;
        }
        return false;
    }

    int16_t getCurrentReferenceFallback() {
        if (last_sent_reference_.has_value()) {
            return *last_sent_reference_;
        }
        if (!hand_) {
            return open_reference_;
        }
        std::vector<int16_t> refs;
        if (hand_->getControlReferences(refs) == 0 && !refs.empty()) {
            return refs.front();
        }
        return open_reference_;
    }

    void onSetClosure(
        const std::shared_ptr<qbsofthand_control::srv::SetClosure::Request> request,
        std::shared_ptr<qbsofthand_control::srv::SetClosure::Response> response) {

        std::lock_guard<std::mutex> lock(mutex_);

        if (!hand_) {
            response->success = false;
            response->message = "device not initialized";
            return;
        }

        const float closure = clampFloat(request->closure, 0.0f, 1.0f);
        const float speed_ratio = clampFloat(request->speed_ratio, 0.01f, 1.0f);
        const float duration_sec = std::max(0.0f, request->duration_sec);

        const int target_int = static_cast<int>(std::lround(
            static_cast<double>(open_reference_) +
            static_cast<double>(close_reference_ - open_reference_) * static_cast<double>(closure)));
        const int16_t target_ref = clampInt16(target_int);

        // 两种控制方式是“分开”的：
        // - duration_sec > 0: 按时长插值到目标（速度由 duration_sec 决定），speed_ratio 不参与。
        // - duration_sec == 0: 按速度比例逐步逼近目标（使用 max_step_per_tick * speed_ratio）。
        motion_.active = true;
        motion_.start_ref = getCurrentReferenceFallback();
        motion_.current_ref = motion_.start_ref;
        motion_.target_ref = target_ref;
        motion_.start_time = now();
        motion_.speed_ratio = speed_ratio;

        if (duration_sec > 0.0f) {
            motion_.mode = ControlMode::Duration;
            motion_.duration = rclcpp::Duration::from_seconds(static_cast<double>(duration_sec));
            response->message = "scheduled (duration mode)";
        } else {
            motion_.mode = ControlMode::Speed;
            motion_.duration = rclcpp::Duration::from_seconds(0.0);
            response->message = "scheduled (speed mode)";
        }

        response->success = true;
    }

    void onActivate(
        const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
        std::shared_ptr<std_srvs::srv::SetBool::Response> response) {

        std::lock_guard<std::mutex> lock(mutex_);
        if (!hand_) {
            response->success = false;
            response->message = "device not initialized";
            return;
        }
        const int ret = hand_->setMotorStates(request->data);
        response->success = (ret == 0);
        response->message = response->success ? "ok" : "failed";
    }

    void onTimer() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!hand_ || !motion_.active) {
            return;
        }

        if (motion_.mode == ControlMode::Speed) {
            const int delta = static_cast<int>(motion_.target_ref) - static_cast<int>(motion_.current_ref);
            const int sign = (delta >= 0) ? 1 : -1;

            const int step = std::max(1, static_cast<int>(std::lround(
                static_cast<double>(max_step_per_tick_) * static_cast<double>(motion_.speed_ratio))));

            if (std::abs(delta) <= step) {
                (void)sendReference(motion_.target_ref);
                motion_.current_ref = motion_.target_ref;
                motion_.active = false;
                return;
            }

            const int next = static_cast<int>(motion_.current_ref) + sign * step;
            const int16_t next_ref = clampInt16(next);
            (void)sendReference(next_ref);
            motion_.current_ref = next_ref;
            return;
        }

        // Duration mode: interpolate based on requested duration.
        const auto t = now();
        if (motion_.duration.nanoseconds() <= 0) {
            (void)sendReference(motion_.target_ref);
            motion_.active = false;
            return;
        }

        const auto elapsed = t - motion_.start_time;
        const double elapsed_s = elapsed.seconds();
        const double total_s = motion_.duration.seconds();

        const double alpha = std::min(1.0, std::max(0.0, elapsed_s / total_s));
        const double ref = static_cast<double>(motion_.start_ref) +
            alpha * static_cast<double>(motion_.target_ref - motion_.start_ref);

        (void)sendReference(clampInt16(static_cast<int>(std::lround(ref))));

        if (alpha >= 1.0) {
            motion_.active = false;
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<QBSoftHandNode>();
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    node->stop();
    node.reset();

    rclcpp::shutdown();
    return 0;
}