/*! \mainpage
 * <img src="qb.svg" align="center" width="200" height="200">
 * <div style="clear: both"></div>
 *
 *  Software License Agreement: BSD 3-Clause License
 *
 *  Copyright (c) 2015-2022, qbrobotics®
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *    following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
 *    following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 *  * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
 *    products derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 *  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef QBROBOTICS_DRIVER_QBROBOTICS_RESEARCH_API_H
#define QBROBOTICS_DRIVER_QBROBOTICS_RESEARCH_API_H

// standard libraries
#include <iomanip>
#include <map>
#include <memory>
#include <thread>
#include <vector>
// project libraries
#include <serial/serial.h>
#include <qbrobotics_research_api/qbrobotics_research_commands.h>

namespace qbrobotics_research_api {

/**
 * @brief Class that handles the communication with 7.X.X firmware devices
 * 
 */
class Communication {
 public:
  /**
   * @brief Structure containing information about the type of device connected
   * 
   */
  struct ConnectedDeviceInfo {
    uint8_t id {0};
    std::string serial_number {};
    std::string type {};
    std::string sub_type {};
  };

  Communication();
  explicit Communication(uint32_t baud_rate);
  explicit Communication(const serial::Serial::Timeout &timeout);
  explicit Communication(uint32_t baud_rate, const serial::Serial::Timeout &timeout);
  virtual ~Communication() = default;

  virtual int listConnectedDevices();
  /**
   * @brief List all the device id connected to the serial port
   * 
   * @param serial_port_name serial port name
   * @param[out] device_ids vector of ConnectedDeviceInfo
   * @return number of devices retrieved
   */
  virtual int listConnectedDevices(const std::string &serial_port_name, std::vector<ConnectedDeviceInfo> &device_ids);
  /**
   * @brief List all the serial ports with a qbrobotics device connected
   * 
   * @param serial_ports_info vector of serial::PortInfo
   * @return number of ports retrieved 
   */
  virtual int listSerialPorts(std::vector<serial::PortInfo> &serial_ports_info);

  /**
   * @brief Close the serial port if it belongs to the opened port set
   * 
   * @param serial_port_name serial port to close
   * @return 0 on success, -1 on error
   */
  virtual int closeSerialPort(const std::string &serial_port_name);

  /**
   * @brief put in the class map \p serial_ports_ a newly created shared pointer Serial class that provides a portable serial port interface. 
   * This method use default baud rate and timeout.
   * 
   * @param serial_port_name
   */
  virtual int createSerialPort(const std::string &serial_port_name);

  /**
   * @brief put in the class map \p serial_ports_ a newly created shared pointer Serial class that provides a portable serial port interface. 
   * This method use default timeout.
   * 
   * @param serial_port_name 
   * @param baud_rate 
   */
  virtual int createSerialPort(const std::string &serial_port_name, uint32_t baud_rate);

  /**
   * @brief put in the class map \p serial_ports_ a newly created shared pointer Serial class that provides a portable serial port interface. 
   * This method use default baud rate.
   * 
   * @param serial_port_name 
   * @param timeout 
   */
  virtual int createSerialPort(const std::string &serial_port_name, const serial::Serial::Timeout &timeout);

  /**
   * @brief put in the class map \p serial_ports_ a newly created shared pointer Serial class that provides a portable serial port interface. 
   * 
   * @param serial_port_name 
   * @param baud_rate
   * @param timeout 
   */
  virtual int createSerialPort(const std::string &serial_port_name, uint32_t baud_rate, const serial::Serial::Timeout &timeout);

  /**
   * Open the serial communication on the given serial port.
   * \param serial_port_name the serial port name
   * \return 0 on success, -1 on error
   */
  virtual int openSerialPort(const std::string &serial_port_name);

  /**
   * Open the serial communication on the given serial port with the default timeout.
   * \param serial_port_name the serial port name
   * \param baud_rate
   * \return 0 on success, -1 on error
   */
  virtual int openSerialPort(const std::string &serial_port_name, uint32_t baud_rate);

  /**
   * Open the serial communication on the given serial port with the default baudate.
   * \param serial_port_name the serial port name
   * \param timeout
   * \return 0 on success, -1 on error
   */
  virtual int openSerialPort(const std::string &serial_port_name, serial::Serial::Timeout &timeout);

  /**
   * Open the serial communication on the given serial port.
   * \param serial_port_name the serial port name
   * \param baud_rate 
   * \param timeout
   * \return 0 on success, -1 on error
   */
  virtual int openSerialPort(const std::string &serial_port_name, uint32_t baud_rate, serial::Serial::Timeout &timeout);

  /*
    -----------------
    parse package, send commands and other utilities to control qbrobotics devices
    -----------------
  */

  virtual int parsePackage(const std::string &serial_port_name, uint8_t device_id, uint8_t command);
  virtual int parsePackage(const std::string &serial_port_name, uint8_t device_id, uint8_t command, std::vector<int8_t> &data_in);
  virtual int sendCommand(const std::string &serial_port_name, uint8_t device_id, uint8_t command);
  virtual int sendCommand(const std::string &serial_port_name, uint8_t device_id, uint8_t command, const std::vector<int8_t> &data_out);
  virtual int sendCommandAndParse(const std::string &serial_port_name, uint8_t device_id, uint8_t command);
  virtual int sendCommandAndParse(const std::string &serial_port_name, uint8_t device_id, uint8_t command, uint8_t max_repeats);
  virtual int sendCommandAndParse(const std::string &serial_port_name, uint8_t device_id, uint8_t command, std::vector<int8_t> &data_in);
  virtual int sendCommandAndParse(const std::string &serial_port_name, uint8_t device_id, uint8_t command, uint8_t max_repeats, std::vector<int8_t> &data_in);
  virtual int sendCommandAndParse(const std::string &serial_port_name, uint8_t device_id, uint8_t command, const std::vector<int8_t> &data_out);
  virtual int sendCommandAndParse(const std::string &serial_port_name, uint8_t device_id, uint8_t command, uint8_t max_repeats, const std::vector<int8_t> &data_out);
  virtual int sendCommandAndParse(const std::string &serial_port_name, uint8_t device_id, uint8_t command, const std::vector<int8_t> &data_out, std::vector<int8_t> &data_in);
  virtual int sendCommandAndParse(const std::string &serial_port_name, uint8_t device_id, uint8_t command, uint8_t max_repeats, const std::vector<int8_t> &data_out, std::vector<int8_t> &data_in);
  virtual int sendCommandBroadcast(const std::string &serial_port_name, uint8_t command);
  virtual int sendCommandBroadcast(const std::string &serial_port_name, uint8_t command, const std::vector<int8_t> &data_out);
  //WARN: broadcast should be used only for non-returning methods (it is unreliable for returning methods)

  virtual int deserializePackage(const std::vector<uint8_t> &package_in, uint8_t &device_id, uint8_t &command);
  virtual int deserializePackage(const std::vector<uint8_t> &package_in, uint8_t &device_id, uint8_t &command, std::vector<int8_t> &data);
  virtual int readPackage(const std::string &serial_port_name, std::vector<uint8_t> &package_in);
  virtual int readPackage(const std::string &serial_port_name, uint8_t &device_id, uint8_t &command);
  virtual int readPackage(const std::string &serial_port_name, uint8_t &device_id, uint8_t &command, std::vector<int8_t> &data);
  virtual int serializePackage(uint8_t device_id, uint8_t command, std::vector<uint8_t> &package_out);
  virtual int serializePackage(uint8_t device_id, uint8_t command, const std::vector<int8_t> &data, std::vector<uint8_t> &package_out);
  virtual int writePackage(const std::string &serial_port_name, const std::vector<uint8_t> &package_out);
  virtual int writePackage(const std::string &serial_port_name, uint8_t device_id, uint8_t command);
  virtual int writePackage(const std::string &serial_port_name, uint8_t device_id, uint8_t command, const std::vector<int8_t> &data);

  virtual uint8_t checksum(const std::vector<uint8_t> &data, uint32_t size);

  template<class T>
  static void swapBytes(T &x) {
    auto a = reinterpret_cast<uint8_t*>(std::addressof(x));
    for(auto b = a+sizeof(T)-1; a<b; ++a,--b) {
      std::swap(*a, *b);
    }
  }
  template<class T>
  static void swapBytes(std::vector<T> &vector) {
    for (auto &&item : vector) {  // auto && is required by std::vector<bool> which returns a prvalueL
      swapBytes(item);
    }
  }
  template<class S, class T>
  static std::vector<S> vectorCast(std::vector<T> vector) {
    auto data_in = reinterpret_cast<S*>(vector.data());
    std::vector<S> data_out(data_in, data_in+vector.size()*sizeof(T)/sizeof(S));
    return data_out;
  }
  template<class S, class T>
  static std::vector<S> vectorCastAndSwap(std::vector<T> vector) {
    auto data_out = vectorCast<S>(vector);
    swapBytes(data_out);
    return data_out;
  }
  template<class S, class T>
  static std::vector<S> vectorSwapAndCast(std::vector<T> vector) {
    swapBytes(vector);
    return vectorCast<S>(vector);
  }

  std::map<std::string, std::shared_ptr<serial::Serial>> getSerialPorts() { return serial_ports_; }

 protected:
  explicit Communication(const Communication &communication);
  explicit Communication(const Communication &communication, const serial::Serial::Timeout &timeout);

  std::map<std::string, std::vector<ConnectedDeviceInfo>> connected_devices_;
  std::map<std::string, serial::PortInfo> serial_ports_info_;
  std::map<std::string, std::shared_ptr<serial::Serial>> serial_ports_;

  uint32_t serial_ports_baud_rate_;
  serial::Serial::Timeout serial_ports_timeout_;

  bool isInSerialPorts(const std::string &serial_port_name);
  bool isInSerialPortsInfo(const std::string &serial_port_name);
};

/**
 * @brief Communication class with a few fix for 6.X.X firmware devices
 * 
 */
class CommunicationLegacy : public Communication {
 public:
  CommunicationLegacy() = default;
  explicit CommunicationLegacy(const Communication &communication);
  explicit CommunicationLegacy(const Communication &communication, const serial::Serial::Timeout &timeout);
  virtual ~CommunicationLegacy() = default;

  int parsePackage(const std::string &serial_port_name, uint8_t device_id, uint8_t command, std::vector<int8_t> &data_in) override;
  int sendCommandAndParse(const std::string &serial_port_name, uint8_t device_id, uint8_t command, uint8_t max_repeats, const std::vector<int8_t> &data_out, std::vector<int8_t> &data_in) override;

  int deserializePackage(const std::vector<uint8_t> &package_in, uint8_t &device_id, uint8_t &command, std::vector<int8_t> &data) override;
  int readLongPackage(const std::string &serial_port_name, std::vector<uint8_t> &package_in);
  int readPackage(const std::string &serial_port_name, uint8_t &device_id, uint8_t &command, std::vector<int8_t> &data) override;
  int readPackage(const std::string &serial_port_name, std::vector<uint8_t> &package_in) override;
};

/**
 * @brief General class that allow to get/set parameters from/to qbrobotics devices
 * 
 */
class Device {
 public:
  /**
   * @brief Manage qbrobotics devices parameters
   * 
   */
  class Params {
   public:
    Params() = default;
    virtual ~Params() = default;

    virtual void initParams(const std::vector<int8_t> &param_buffer);

    template<class T>
    static void getParameter(uint8_t param_id, const std::vector<int8_t> &param_buffer, std::vector<T> &param_vector) {
      auto number_of_values = static_cast<uint8_t>(param_buffer.at((param_id-1)*PARAM_BYTE_SLOT + 2));
      auto param_iter = param_buffer.begin() + (param_id-1)*PARAM_BYTE_SLOT + 3;
      std::vector<int8_t> param_bytes(param_iter, param_iter + number_of_values*sizeof(T));
      param_vector = Communication::vectorCastAndSwap<T>(param_bytes);
    }
    template<class T>
    static void getParameter(uint8_t param_id, const std::vector<int8_t> &param_buffer, T &param) {
      std::vector<T> param_vector;
      getParameter(param_id, param_buffer, param_vector);
      param = param_vector.front();
    }

    std::string serial_number {};
    std::string type {};
    std::string sub_type {};

    uint8_t id {0};
    std::vector<float> position_pid;
    std::vector<float> current_pid;
    uint8_t startup_activation {false};  // should be bool
    uint8_t input_mode {0};
    uint8_t control_mode {0};
    std::vector<uint8_t> encoder_resolutions;
    std::vector<int16_t> encoder_offsets;
    std::vector<float> encoder_multipliers;
    uint8_t use_position_limits {false};  // should be bool
    std::vector<int32_t> position_limits;
    std::vector<int32_t> position_max_steps;
    int16_t current_limit {0};
    uint8_t rate_limiter {0};  // param id after softhand params
  };

  explicit Device(std::shared_ptr<Communication> communication, std::string name, std::string serial_port, uint8_t id);
  explicit Device(std::shared_ptr<Communication> communication, std::string name, std::string serial_port, uint8_t id, bool init_params);
  explicit Device(std::shared_ptr<Communication> communication, std::string name, std::string serial_port, uint8_t id, bool init_params, std::unique_ptr<Params> params);
  virtual ~Device() = default;

  virtual int ping();

  /**
   * @brief Get the reference command sent to the motor(s) of the device.
   * 
   * @param[out] control_references[tick] vector passed by reference
   * @return 0 on success, -1 on error
   */
  virtual int getControlReferences(std::vector<int16_t> &control_references);

  /**
   * @brief Get the device current(s) absorbed by the motor(s)
   * 
   * @param[out] currents[mA] vector passed by reference
   * @return 0 on success, -1 on error
   */
  virtual int getCurrents(std::vector<int16_t> &currents);

  /**
   * @brief Get the device currents absorbed by the motor(s) and its/their position(s)
   * 
   * @param[out] currents[mA] vector passed by reference
   * @param[out] positions[tick] vector passed by reference
   * @return 0 on success, -1 on error
   */
  virtual int getCurrentsAndPositions(std::vector<int16_t> &currents, std::vector<int16_t> &positions);

  /**
   * @brief Get the Positions given by the encoders mounted on the device
   * 
   * @param[out] positions[tick] vector passed by reference
   * @return 0 on success, -1 on error
   */
  virtual int getPositions(std::vector<int16_t> &positions);

  /**
   * @brief Get the motor(s) velocitie(s)
   * 
   * @param[out] velocities[tick/s] vector passed by reference
   * @return 0 on success, -1 on error
   */
  virtual int getVelocities(std::vector<int16_t> &velocities);

  /**
   * @brief Get the motor(s) acceleration(s)
   * 
   * @param[out] accelerations[tick/s^2] vector passed by reference
   * @return 0 on success, -1 on error
   */
  virtual int getAccelerations(std::vector<int16_t> &accelerations);

  /**
   * @brief Get the device motor state
   * 
   * @param[out] motor_state boolean passed by reference
   * @return 0 on success, -1 on error
   */
  virtual int getMotorStates(bool &motor_state);
  virtual int getCycleTime(int16_t &cycle_time);

  /**
   * @brief Set motor(s) control reference - position[tick]
   * 
   * @param control_references[tick]
   * @return 0 on success, -1 on error
   */
  virtual int setControlReferences(const std::vector<int16_t> &control_references);

  /**
   * @brief Set motor(s) control reference and wait for acknowledge - position[tick](only for legacy qbmoves)
   * 
   * @param control_references[tick]
   * @return 0 on success, -1 on error
   */
  virtual int setControlReferencesAndWait(const std::vector<int16_t> &control_references);

  /**
   * @brief Activate or deactivate the motor(s)
   * 
   * @param motor_state the desired motor(s) state(s)
   * @return 0 on success, -1 on error 
   */
  virtual int setMotorStates(bool motor_state);

  /**
   * @brief Get all system information
   * 
   * @param[out] info std::string passed by reference
   * @return 0 on success, -1 on error 
   */
  virtual int getInfo(std::string &info);

  /**
   * @brief Get the Info of the device
   * 
   * @param info_type - 0 All system information; - 1 Cycles information; - 2 Read Firmware Parameters from SD file; - 3 Read Usage Data from SD file.
   * @param info[out] std::string passed by reference
   * @return 0 on success, -1 on error 
   */
  virtual int getInfo(uint16_t info_type, std::string &info);

  /**
   * @brief Get the parameters from the device
   * 
   * @param[out] param_buffer vector passed by reference
   * @return 0 on success, -1 on error 
   */
  virtual int getParameters(std::vector<int8_t> &param_buffer);

  /**
   * @brief Get the parameters from the device with ID \p id
   * 
   * @param[out] id device ID
   * @param[out] param_buffer vector passed by reference
   * @return 0 on success, -1 on error 
   * 
   * \sa getParameters(uint8_t id, std::vector<int8_t> &param_buffer)
   */
  virtual int getParameters(uint8_t id, std::vector<int8_t> &param_buffer);

  /**
   * @brief Update the device ID in the class variable \p param_
   *
   * @return 0 on success, -1 on error 
   */
  virtual int getParamId();

  /**
   * @brief Get the device ID
   * 
   * @param[out] id device ID
   * @return 0 on success, -1 on error 
   * 
   * \sa getParameters(uint8_t id, std::vector<int8_t> &param_buffer)
   */
  virtual int getParamId(uint8_t &id);

  /**
   * @brief Update the device position PID parameters in the class variable \p param_
   *
   * @return 0 on success, -1 on error 
   */
  virtual int getParamPositionPID();

  /**
   * @brief Get the device position PID parameters
   *
   * @param[out] position_pid
   * @return 0 on success, -1 on error 
   */
  virtual int getParamPositionPID(std::vector<float> &position_pid);

  /**
   * @brief Update the device current PID parameters in the class variable \p param_
   *
   * @return 0 on success, -1 on error 
   */
  virtual int getParamCurrentPID();

  /**
   * @brief Get the device current PID parameters
   *
   * @param[out] current_pid
   * @return 0 on success, -1 on error 
   */
  virtual int getParamCurrentPID(std::vector<float> &current_pid);

  /**
   * @brief Update the startup activation parameter in the class variable \p param_
   *
   * @return 0 on success, -1 on error 
   */
  virtual int getParamStartupActivation();

  /**
   * @brief Get the startup activation parameter
   *
   * @param[out] startup_activation
   * @return 0 on success, -1 on error 
   */
  virtual int getParamStartupActivation(uint8_t &startup_activation);

  /**
   * @brief Get the input mode parameter in the class variable \p param_. These values are:
   * - 0 - References through external commands 
   * - 1 - Encoder 3 drives all inputs
   * - 2 - Use EMG measure to proportionally drive the position of the motor 1
   * - 3 - Use 2 EMG signals to drive motor position
   * - 4 - Use 2 EMG. First reaching threshold wins and its value defines hand closure
   * - 5 - Use 2 EMG. First reaching threshold wins and its value defines hand closure. Wait for both EMG to lower under threshold
   * 
   * @return 0 on success, -1 on error 
   */
  virtual int getParamInputMode();

  /**
   * @brief Get the input parameter to the correspective value of the input mode. 
   * @param[out] input_mode The possible values are:
   * - 0 - References through external commands 
   * - 1 - Encoder 3 drives all inputs
   * - 2 - Use EMG measure to proportionally drive the position of the motor 1
   * - 3 - Use 2 EMG signals to drive motor position
   * - 4 - Use 2 EMG. First reaching threshold wins and its value defines hand closure
   * - 5 - Use 2 EMG. First reaching threshold wins and its value defines hand closure. Wait for both EMG to lower under threshold
   * 
   * @return 0 on success, -1 on error 
   */
  virtual int getParamInputMode(uint8_t &input_mode);

  /**
   * @brief Update the device control mode in the class variable \p param_
   * - 0 - Classic position control
   * - 1 - Direct PWM value
   * - 2 - Current control
   * - 3 - Position and current control
   * - 4 - Deflection control (only for qbmoves)
   * - 5 - Deflection and current control (only for qbmoves)
   * @return 0 on success, -1 on error 
   */
  virtual int getParamControlMode();

  /**
   * @brief Get the control mode parameter
   *
   * @param[out] control_mode
   * - 0 - Classic position control
   * - 1 - Direct PWM value
   * - 2 - Current control
   * - 3 - Position and current control
   * - 4 - Deflection control (only for qbmoves)
   * - 5 - Deflection and current control (only for qbmoves)
   * @return 0 on success, -1 on error 
   */
  virtual int getParamControlMode(uint8_t &control_mode);

  /**
   * @brief Update the device encoder resolutions in the class variable \p param_
   *
   * @return 0 on success, -1 on error 
   */
  virtual int getParamEncoderResolutions();

  /**
   * @brief Get the encoder resolution parameters
   *
   * @param[out] encoder_resolutions
   * @return 0 on success, -1 on error 
   */
  virtual int getParamEncoderResolutions(std::vector<uint8_t> &encoder_resolutions);

  /**
   * @brief Update the device encoder offsets in the class variable \p param_
   *
   * @return 0 on success, -1 on error 
   */
  virtual int getParamEncoderOffsets();

  /**
   * @brief Get the encoder offsets parameters
   *
   * @param[out] encoder_offsets
   * @return 0 on success, -1 on error 
   */
  virtual int getParamEncoderOffsets(std::vector<int16_t> &encoder_offsets);

  /**
   * @brief Update the device encoder multipliers in the class variable \p param_
   *
   * @return 0 on success, -1 on error 
   */
  virtual int getParamEncoderMultipliers();

  /**
   * @brief Get the encoder multipliers parameters
   *
   * @param[out] encoder_multipliers
   * @return 0 on success, -1 on error 
   */
  virtual int getParamEncoderMultipliers(std::vector<float> &encoder_multipliers);

  /**
   * @brief Update the the use of position limits in the class variable \p param_
   *
   * @return 0 on success, -1 on error 
   */
  virtual int getParamUsePositionLimits();

  /**
   * @brief Get the use of position limits parameter
   *
   * @param[out] use_position_limits 0 false, 1 true
   * @return 0 on success, -1 on error 
   */
  virtual int getParamUsePositionLimits(uint8_t &use_position_limits);

  /**
   * @brief Update the position limits parameters in the class variable \p param_
   *
   * @return 0 on success, -1 on error 
   */
  virtual int getParamPositionLimits();

  /**
   * @brief Get position limits parameters
   *
   * @param[out] position_limits
   * @return 0 on success, -1 on error 
   */
  virtual int getParamPositionLimits(std::vector<int32_t> &position_limits);

  /**
   * @brief Update the max step position parameters in the class variable \p param_
   *
   * @return 0 on success, -1 on error 
   */
  virtual int getParamPositionMaxSteps();

  /**
   * @brief Get max steps position parameters
   *
   * @param[out] position_max_steps
   * @return 0 on success, -1 on error 
   */
  virtual int getParamPositionMaxSteps(std::vector<int32_t> &position_max_steps);

  /**
   * @brief Update the current limits parameters in the class variable \p param_
   *
   * @return 0 on success, -1 on error 
   */
  virtual int getParamCurrentLimit();

  /**
   * @brief Get current limit parameter
   *
   * @param[out] current_limit[mA]
   * @return 0 on success, -1 on error 
   */
  virtual int getParamCurrentLimit(int16_t &current_limit);

  /**
   * @brief Set the Parameter specified by \p param_type with the values stored in \p param_data
   * 
   * @param param_type ID of the parameter to be set
   * @param param_data value(s) to be set
   * @return 0 on success, < 0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParameter(uint16_t param_type, const std::vector<int8_t> &param_data);

  /**
   * @brief Set the ID of the device
   * 
   * @param id desired device ID
   * @return 0 on success, < 0 on error  
   */
  virtual int setParamId(uint8_t id);

  /**
   * @brief Set the position PID parameters of the device
   * 
   * @param position_pid desired device PID(position) parameters
   * @return 0 on success, < 0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamPositionPID(const std::vector<float> &position_pid);

  /**
   * @brief Set the current PID parameters of the device
   * 
   * @param current_pid desired device PID(current) parameters
   * @return 0 on success, < 0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamCurrentPID(const std::vector<float> &current_pid);

  /**
   * @brief Set the startup activation parameter of the device
   * 
   * @param startup_activation
   * @return 0 on success, < 0 on error  
   */
  virtual int setParamStartupActivation(bool startup_activation);

  /**
   * @brief Set the input mode parameter of the device
   * 
   * @param input_mode
   * @return 0 on success, < 0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamInputMode(uint8_t input_mode);

  /**
   * @brief Set the control mode parameter of the device
   * 
   * @param control_mode
   * @return 0 on success, < 0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamControlMode(uint8_t control_mode);

  /**
   * @brief Set the encoder resolutions parameters of the device
   * 
   * @param encoder_resolutions
   * @return 0 on success, < 0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamEncoderResolutions(const std::vector<uint8_t> &encoder_resolutions);

  /**
   * @brief Set the encoder offsets parameters of the device
   * 
   * @param encoder_offsets
   * @return 0 on success, < 0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamEncoderOffsets(const std::vector<int16_t> &encoder_offsets);

  /**
   * @brief Set the encoder multipliers parameters of the device
   * 
   * @param encoder_multipliers
   * @return 0 on success, < 0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamEncoderMultipliers(const std::vector<float> &encoder_multipliers);

  /**
   * @brief Enable or disable the use of position limits
   * 
   * @param use_position_limits
   * @return 0 on success, < 0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamUsePositionLimits(bool use_position_limits);

  /**
   * @brief Set the position limits parameters of the device
   * 
   * @param position_limits
   * @return 0 on success, < 0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamPositionLimits(const std::vector<int32_t> &position_limits);

  /**
   * @brief Set the position max steps parameters of the device
   * 
   * @param position_max_steps
   * @return 0 on success, < 0 on error
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamPositionMaxSteps(const std::vector<int32_t> &position_max_steps);

  /**
   * @brief Set the cerrent limit parameter of the device
   * 
   * @param current_limit
   * @return 0 on success, < 0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamCurrentLimit(int16_t current_limit);

  /**
   * @brief Set motor(s) zero(s) to actual encoder reading
   * @return 0 on success, -1 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamZeros();

  /**
   * @brief Set the device baud rate
   * 
   * @param prescaler_divider
   * @return 0 on success, -1 on error
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setParamBaudrate(uint8_t prescaler_divider);

  /**
   * @brief Restore factory parameter on 7.X.X firmware devices. For 6.X.X it executes INIT_MEM command
   * 
   * @return 0 on success, <0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int restoreFactoryDataMemory();

  /**
   * @brief Restore user parameter on 7.X.X firmware devices. 
   * \warning not use it in 6.X.X devices. 
   * 
   * @return 0 on success, <0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int restoreUserDataMemory();

  /**
   * @brief Store the changed parameters on 7.X.X firmware devices in factory memory. 
   * \warning not use it in 6.X.X devices. 
   * 
   * @return 0 on success, <0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int storeFactoryDataMemory();

  /**
   * @brief Store the changed parameters on 7.X.X firmware devices in user memory. 
   * \warning not use it in 6.X.X devices. 
   * 
   * @return 0 on success, <0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int storeUserDataMemory();

  /**
   * @brief Set the bootloader mode
   * 
   * @return 0 on success, <0 on error  
   * @warning The improper use of this function could damage the device and invalidate the device warranty. Contact our support team (support@qbrobotics.com) for more information.
   */
  virtual int setBootloaderMode();

  std::shared_ptr<Params> getParams() { return params_; }
  bool isQbmove() { return params_->type == "001"; }
  bool isSHPRO()  { return params_->type == "006" && params_->sub_type == "001"; }
  bool isSH2M() { return params_->type == "006" && params_->sub_type == "020"; }

 protected:
  std::string name_;
  std::string serial_port_;
  std::shared_ptr<Params> params_;
  std::shared_ptr<Communication> communication_;
};

}  // namespace qbrobotics_research_api

#endif  // QBROBOTICS_DRIVER_QBROBOTICS_RESEARCH_API_H
