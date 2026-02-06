/***
 *  Software License Agreement: BSD 3-Clause License
 *
 *  Copyright (c) 2022, qbrobotics®
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

#include <stdlib.h>

#include <iostream>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <memory>
#include <set>
#include <serial/serial.h>
#include <qbrobotics_research_api/qbsofthand_research_api.h>

// handler to manage the communication with qbdevices
std::shared_ptr<qbrobotics_research_api::Communication> communication_handler_;
// Communication ports
std::vector<serial::PortInfo> serial_ports_;
std::map<int, std::shared_ptr<qbrobotics_research_api::qbSoftHandLegacyResearch>> soft_hands_;
// IDs of connected devices 
std::vector<qbrobotics_research_api::Communication::ConnectedDeviceInfo> device_ids_;

int open(const std::string &serial_port) {
  if (!std::regex_match(serial_port, std::regex("/dev/ttyUSB[[:digit:]]+"))) {
    return -1;
  }
  if(communication_handler_->openSerialPort(serial_port) < 0){
    std::cerr << "Not able to open: " << serial_port << " serial port";
    return -1;
  }
  std::cout << "Opened: " << serial_port << " serial port"<< std::endl;
  return 0;
}

// Scan ports for qbrobotics devices
int scanForDevices(const int &max_repeats) {
    communication_handler_ = std::make_shared<qbrobotics_research_api::CommunicationLegacy>(); // make shared pointer that handles the communication
  if (communication_handler_->listSerialPorts(serial_ports_) < 0) {
      std::cerr << "[scanForDevices] no serial ports found" << std::endl; 
      return -1;
  }
  int qbrobotics_devices_found = 0;
  for(auto &serial_port:serial_ports_){ // scan and open all the serial port
    int failures = 0;
    while (failures <= max_repeats) {
      if (open(serial_port.serial_port) != 0) {
        failures++;
        continue;
      }
      break;
    }
    if (failures >= max_repeats) {
      continue;
    }

    if (communication_handler_->listConnectedDevices(serial_port.serial_port, device_ids_) >= 0) { // retrieved at least a qbrobotics device
      for(auto &device_id:device_ids_) {
        if (device_id.id == 120 || device_id.id == 0) {
          std::cout << "Not valid device retrieved!" << std::endl;
          continue;  // ID 120 is reserved, ID 0 is for sure an error
        }
        soft_hands_.insert(std::make_pair(static_cast<int>(device_id.id), std::make_shared<qbrobotics_research_api::qbSoftHandLegacyResearch>(communication_handler_, "dev", serial_port.serial_port, device_id.id)));
        qbrobotics_devices_found++;
      }
      if (qbrobotics_devices_found == 0) {
        std::cerr << "[scanForDevices] no qbrobotics devices found" << std::endl; 
      }
    }
  }
  return qbrobotics_devices_found;
}


int main() {
  std::cout << "qbrobotics devices found: " << scanForDevices(2) << std::endl;
  for (auto &id:device_ids_){
    if (id.id == 120 || id.id == 0) {
      continue;  // ID 120 is reserved, ID 0 is for sure an error
    }
    std::string info_string;
    soft_hands_.at((int)id.id)->getInfo(INFO_ALL, info_string);
    std::cout << info_string << std::endl << "----" << std::endl;

    std::cout << "[getControlReferences()] ";                                     //getControlReferences()
    std::vector<int16_t> control_references;
    soft_hands_.at((int)id.id)->getControlReferences(control_references);
    for (auto &reference:control_references){
      std::cout << reference << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getCurrents()] ";                                              //getCurrents()
    std::vector<int16_t> currents;
    soft_hands_.at((int)id.id)->getCurrents(currents);
    for (auto &current:currents){
      std::cout << current << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getCurrentsAndPositions()] ";                                  //getCurrentsAndPositions()
    std::vector<int16_t> positions;
    soft_hands_.at((int)id.id)->getCurrentsAndPositions(currents, positions);
    for (auto &current:currents){
      std::cout << current << " ";
    }
    for (auto &position:positions){
      std::cout << position << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getPositions()] ";                                             //getPositions()
    soft_hands_.at((int)id.id)->getPositions(positions);
    for (auto &position:positions){
      std::cout << position << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getVelocities()] ";                                            //getVelocities()
    std::vector<int16_t> velocities;
    soft_hands_.at((int)id.id)->getVelocities(velocities);
    for (auto &velocity:velocities){
      std::cout << velocity << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getAccelerations()] ";                                         //getAccelerations()
    std::vector<int16_t> accelerations;
    soft_hands_.at((int)id.id)->getAccelerations(accelerations);
    for (auto &acceleration:accelerations){
      std::cout << acceleration << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[setMotorStates()] ";                                           //setMotorStates()
    bool activate = true;
    if (soft_hands_.at((int)id.id)->setMotorStates(activate) == 0){
      std::cout << "Motors are active";
    } else {
      std::cout << "Something went wrong while activating motors";
      break;
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[setControlReferences()] ";                                     //setControlReferences()
    control_references = {10000};
    soft_hands_.at((int)id.id)->setControlReferences(control_references);
    for (auto &control_reference:control_references){
      std::cout << control_reference << " ";
    }
    std::cout << "\n----"<< std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "[setControlReferences()] ";                                     //setControlReferences()
    control_references = {19000};
    soft_hands_.at((int)id.id)->setControlReferences(control_references);
    for (auto &control_reference:control_references){
      std::cout << control_reference << " ";
    }
    std::cout << "\n----"<< std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "[setControlReferences()] ";                                     //setControlReferences()
    control_references = {0, 0};
    soft_hands_.at((int)id.id)->setControlReferences(control_references);
    for (auto &control_reference:control_references){
      std::cout << control_reference << " ";
    }
    std::cout << "\n----"<< std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "[getMotorStates()] ";                                           //getMotorStates()
    soft_hands_.at((int)id.id)->getMotorStates(activate);
    if(activate){
      std::cout << "Motors are active";
    } else {
      std::cout << "Motors are not active";
      break;
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamId()] ";                                               //getParamId()
    uint8_t device_id;
    soft_hands_.at((int)id.id)->getParamId(device_id);
    std::cout << (int)device_id << " ";
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamPositionPID()] ";                                      //getParamPositionPID()
    std::vector<float> PID;
    soft_hands_.at((int)id.id)->getParamPositionPID(PID);
    for (auto &param:PID){
      std::cout << param << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamCurrentPID()] ";                                      //getParamCurrentPID()
    soft_hands_.at((int)id.id)->getParamCurrentPID(PID);
    for (auto &param:PID){
      std::cout << param << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamStartupActivation()] ";                               //getParamStartupActivation()
    uint8_t activation;
    soft_hands_.at((int)id.id)->getParamStartupActivation(activation);
    std::cout << (int)activation;
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamInputMode()] ";                                       //getParamInputMode()
    uint8_t input_mode;
    soft_hands_.at((int)id.id)->getParamInputMode(input_mode);
    std::cout << (int)input_mode;
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamControlMode()] ";                                     //getParamControlMode()
    uint8_t control_mode;
    soft_hands_.at((int)id.id)->getParamControlMode(control_mode);
    std::cout << (int)control_mode;
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamEncoderResolutions()] ";                              //getParamEncoderResolutions()
    std::vector<uint8_t> encoder_resolutions;
    soft_hands_.at((int)id.id)->getParamEncoderResolutions(encoder_resolutions);
    for (auto &param:encoder_resolutions){
      std::cout << (int)param << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamEncoderOffsets()] ";                                   //getParamEncoderOffsets()
    std::vector<int16_t> encoder_offsets;                                         
    soft_hands_.at((int)id.id)->getParamEncoderOffsets(encoder_offsets);
    for (auto &offset:encoder_offsets){
      std::cout << offset << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamEncoderMultipliers()] ";                              //getParamEncoderMultipliers()
    std::vector<float> encoder_multipliers;
    soft_hands_.at((int)id.id)->getParamEncoderMultipliers(encoder_multipliers);
    for (auto &param:encoder_multipliers){
      std::cout << param << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamUsePositionLimits()] ";                               //getParamUsePositionLimits()
    uint8_t use_position_limits;
    soft_hands_.at((int)id.id)->getParamUsePositionLimits(use_position_limits);
    std::cout << (int)use_position_limits;
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamPositionLimits()] ";                                  //getParamPositionLimits()
    std::vector<int32_t> position_limits;
    soft_hands_.at((int)id.id)->getParamPositionLimits(position_limits);
    for (auto &param:position_limits){
      std::cout << (int)param << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamPositionMaxSteps()] ";                                //getParamPositionMaxSteps()
    std::vector<int32_t> position_max_steps;
    soft_hands_.at((int)id.id)->getParamPositionMaxSteps(position_max_steps);
    for (auto &param:position_max_steps){
      std::cout << (int)param << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[getParamCurrentLimit()] ";                                    //getParamCurrentLimit()
    int16_t current_limit;
    soft_hands_.at((int)id.id)->getParamCurrentLimit(current_limit);
    std::cout << (int)current_limit << " ";
    std::cout << "\n--------------\n"<< std::endl;

    std::cout << "[getPositions()] ";                                                //getPositions()
    soft_hands_.at((int)id.id)->getPositions(positions);
    for (auto &position:positions){
      std::cout << position << " ";
    }
    std::cout << "\n----"<< std::endl;

    std::cout << "[setMotorStates()] ";                                              //setMotorStates()
    activate = false;
    if (soft_hands_.at((int)id.id)->setMotorStates(activate) == 0){
      std::cout << "Motors are not active";
    } else {
      std::cout << "Something went wrong while deactivating motors";
    }
    std::cout << "\n----"<< std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  
  // Close serial port
  for (auto &port:serial_ports_){
    if(communication_handler_->closeSerialPort(port.serial_port) == 0){
      std::cout << "serial port "<< port.serial_port <<  " closed" << std::endl;
    }
  }
  
}
