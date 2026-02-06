/***
 *  Software License Agreement: BSD 3-Clause License
 *
 *  Copyright (c) 2015-2021, qbrobotics®
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

#ifndef QBROBOTICS_DRIVER_QBSOFTHAND_RESEARCH_API_H
#define QBROBOTICS_DRIVER_QBSOFTHAND_RESEARCH_API_H

#include <qbrobotics_research_api/qbrobotics_research_api.h>

namespace qbrobotics_research_api {

class qbSoftHandResearch : public Device {
 public:
  class Params : public Device::Params {
   public:
    Params() = default;
    ~Params() override = default;

    void initParams(const std::vector<int8_t> &param_buffer) override;

    uint8_t use_emg_calibration {false};
    std::vector<uint16_t> emg_thresholds;
    std::vector<uint32_t> emg_max_value;
    uint8_t emg_speed {0};
    uint8_t use_double_encoder {false};
    int8_t handle_ratio {0};
    uint8_t use_pwm_rescaling {false};
    std::vector<float> current_lookup_table;
  };

  explicit qbSoftHandResearch(std::shared_ptr<Communication> communication, std::string name, std::string serial_port, uint8_t id);
  explicit qbSoftHandResearch(std::shared_ptr<Communication> communication, std::string name, std::string serial_port, uint8_t id, bool init_params);
  explicit qbSoftHandResearch(std::shared_ptr<Communication> communication, std::string name, std::string serial_port, uint8_t id, bool init_params, std::unique_ptr<Device::Params> params);
  ~qbSoftHandResearch() override = default;

  int openCloseCycles(uint16_t speed, uint16_t cycles);
  int getEMGs(std::vector<int16_t> &emg_values);
  int getJoystick(std::vector<int16_t> &joystick_values);

  int getParamUseEMGCalibration();
  int getParamUseEMGCalibration(uint8_t &use_emg_calibration);
  int getParamEMGThresholds();
  int getParamEMGThresholds(std::vector<uint16_t> &emg_thresholds);
  int getParamEMGMaxValues();
  int getParamEMGMaxValues(std::vector<uint32_t> &emg_max_value);
  int getParamEMGSpeed();
  int getParamEMGSpeed(uint8_t &emg_speed);
  int getParamUseDoubleEncoder();
  int getParamUseDoubleEncoder(uint8_t &use_double_encoder);
  int getParamHandleRatio();
  int getParamHandleRatio(int8_t &handle_ratio);
  int getParamUsePWMRescaling();
  int getParamUsePWMRescaling(uint8_t &use_pwm_rescaling);
  int getParamCurrentLookupTable();
  int getParamCurrentLookupTable(std::vector<float> &current_lookup_table);
  int getParamRateLimiter();
  int getParamRateLimiter(uint8_t &rate_limiter);

  int setParamUseEMGCalibration(uint8_t use_emg_calibration);
  int setParamEMGThresholds(const std::vector<uint16_t> &emg_thresholds);
  int setParamEMGMaxValues(const std::vector<uint32_t> &emg_max_value);
  int setParamEMGSpeed(uint8_t emg_speed);
  int setParamUseDoubleEncoder(uint8_t use_double_encoder);
  int setParamHandleRatio(int8_t handle_ratio);
  int setParamUsePWMRescaling(uint8_t use_pwm_rescaling);
  int setParamCurrentLookupTable(const std::vector<float> &current_lookup_table);
  int setParamRateLimiter(uint8_t rate_limiter);
};

class qbSoftHandLegacyResearch : public qbSoftHandResearch {
 public:
  class Params : public qbSoftHandResearch::Params {
   public:
    Params() = default;
    ~Params() override = default;

//    void initParams(const std::vector<int8_t> &param_buffer) override;  // not necessary at the moment
  };

  explicit qbSoftHandLegacyResearch(std::shared_ptr<Communication> communication, std::string name, std::string serial_port, uint8_t id);
  explicit qbSoftHandLegacyResearch(std::shared_ptr<Communication> communication, std::string name, std::string serial_port, uint8_t id, bool init_params);
  explicit qbSoftHandLegacyResearch(std::shared_ptr<Communication> communication, std::string name, std::string serial_port, uint8_t id, bool init_params, std::unique_ptr<Device::Params> params);
  ~qbSoftHandLegacyResearch() override = default;

  // same as qbmoveOldResearch
  int setMotorStates(bool motor_state) override;

  int setParameter(uint16_t param_type, const std::vector<int8_t> &param_data) override;
  int setParamId(uint8_t id) override;
  int setParamZeros() override;
};

}  // namespace qbrobotics_research_api

#endif  // QBROBOTICS_DRIVER_QBSOFTHAND_RESEARCH_API_H
