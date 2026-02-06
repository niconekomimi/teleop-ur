/***
 *  MIT License
 *
 *  Copyright (c) 2020 Alessandro Tondo
 *  Copyright (c) 2012 William Woodall, John Harrison
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 *  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 *  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 *  to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
 *  the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 *  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 *  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

#ifndef SERIAL_H
#define SERIAL_H

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace serial {

/***
 *  Enumeration defines the possible bytesizes for the serial port.
 */
typedef enum {
  fivebits = 5,
  sixbits = 6,
  sevenbits = 7,
  eightbits = 8
} bytesize_t;

/***
 *  Enumeration defines the possible parity types for the serial port.
 */
typedef enum {
  parity_none = 0,
  parity_odd = 1,
  parity_even = 2,
  parity_mark = 3,
  parity_space = 4
} parity_t;

/***
 *  Enumeration defines the possible stopbit types for the serial port.
 */
typedef enum {
  stopbits_one = 1,
  stopbits_two = 2,
  stopbits_one_point_five
} stopbits_t;

/***
 *  Enumeration defines the possible flowcontrol types for the serial port.
 */
typedef enum {
  flowcontrol_none = 0,
  flowcontrol_software,
  flowcontrol_hardware
} flowcontrol_t;

/***
 *  Class that provides a portable serial port interface.
 */
class Serial {
 public:
  class Timeout {
   public:
    /***
     *  Create a non-blocking communication mode (since the Timeout structure is filled with 0).
     */
    Timeout() = default;

    /***
     *  Create a simplified constant-time Timeout structure (@sa Timeout below).
     *  @param read_write_constant A constant used to calculate the total timeout period for read/write operations.
     */
    explicit Timeout(uint32_t read_write_constant)
        // the ugly parentheses around `(std::max)()` are required on Windows (this is to avoid to `#undef max` or to `#define NOMINMAX`)
        : Timeout((std::numeric_limits<uint32_t>::max)(), read_write_constant, 0, read_write_constant, 0) {}

    /***
     *  Create a Timeout structure where all the parameters are expressed in milliseconds and refer to the documentation
     *  of @p COMMTIMEOUTS (https://docs.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-commtimeouts).@n
     *  @param inter_byte The maximum time allowed to elapse before the arrival of the next byte.
     *  @param read_constant A constant used to calculate the total timeout period for read operations.
     *  @param read_multiplier The multiplier used to calculate the total timeout period for read operations.
     *  @param write_constant A constant used to calculate the total timeout period for write operations.
     *  @param write_multiplier The multiplier used to calculate the total timeout period for write operations.
     */
    explicit Timeout(uint32_t inter_byte, uint32_t read_constant, uint32_t read_multiplier, uint32_t write_constant, uint32_t write_multiplier)
        : inter_byte_(inter_byte),
          read_constant_(read_constant),
          read_multiplier_(read_multiplier),
          write_constant_(write_constant),
          write_multiplier_(write_multiplier) {}

    ~Timeout() = default;

    std::chrono::milliseconds getInterByte() { return inter_byte_; }
    uint32_t getInterByteMilliseconds() { return inter_byte_.count(); }
    std::chrono::milliseconds getReadConstant() { return read_constant_; }
    uint32_t getReadConstantMilliseconds() { return read_constant_.count(); }
    std::chrono::milliseconds getReadMultiplier() { return read_multiplier_; }
    uint32_t getReadMultiplierMilliseconds() { return read_multiplier_.count(); }
    std::chrono::milliseconds getWriteConstant() { return write_constant_; }
    uint32_t getWriteConstantMilliseconds() { return write_constant_.count(); }
    std::chrono::milliseconds getWriteMultiplier() { return write_multiplier_; }
    uint32_t getWriteMultiplierMilliseconds() { return write_multiplier_.count(); }

    /***
     *  @param size The byte size expected to be read.
     *  @return The next timeout deadline for the current read settings and the given byte size.
     */
    std::chrono::steady_clock::time_point getReadDeadline(size_t size = 0) {
      return std::chrono::steady_clock::now() + read_constant_ + read_multiplier_*size;
    }

    /***
     *  @param size The byte size expected to be written.
     *  @return The next timeout deadline for the current write settings and the given byte size.
     */
    std::chrono::steady_clock::time_point getWriteDeadline(size_t size = 0) {
      return std::chrono::steady_clock::now() + write_constant_ + write_multiplier_*size;
    }

    /***
     *  @param deadline The read/write timeout deadline.
     *  @return The remaining microseconds w.r.t. the given deadline.
     *  @sa getReadDeadline, getWriteDeadline
     */
    static std::chrono::microseconds remainingMicroseconds(std::chrono::steady_clock::time_point deadline) {
      return std::chrono::duration_cast<std::chrono::microseconds>(deadline - std::chrono::steady_clock::now());
    }

   private:
    std::chrono::duration<uint32_t, std::milli> inter_byte_;
    std::chrono::duration<uint32_t, std::milli> read_constant_;
    std::chrono::duration<uint32_t, std::milli> read_multiplier_;
    std::chrono::duration<uint32_t, std::milli> write_constant_;
    std::chrono::duration<uint32_t, std::milli> write_multiplier_;
  };

  /***
   *  Create a Serial object and open the serial port if its name is provided. If not, it remains closed until
   *  @p Serial::open is called.
   *  @param port_name The path of the serial port, e.g. 'COM1' on Windows and '/dev/ttyS0' on Linux.
   *  @param baudrate The baud rate of the serial communication.
   *  @param timeout The timeout conditions of the serial communication. @sa Serial::Timeout
   *  @param bytesize The payload size  of the serial communication (default is @p eightbits). @sa bytesize_t
   *  @param parity The parity method of the serial communication (default is @p parity_none). @sa parity_t
   *  @param stopbits The number of stop bits of the serial communication (default is @p stopbits_one). @sa stopbits_t
   *  @param flowcontrol The type of flow control of the serial communication (default is @p flowcontrol_none). @sa flowcontrol_t
   *  @throw serial::SerialIOException
   *  @throw serial::SerialInvalidArgumentException
   */
  explicit Serial(const std::string &port_name = "", uint32_t baudrate = 9600, Timeout timeout = Timeout(),
                  bytesize_t bytesize = eightbits, parity_t parity = parity_none, stopbits_t stopbits = stopbits_one,
                  flowcontrol_t flowcontrol = flowcontrol_none);

  virtual ~Serial();

  Serial(const Serial &) = delete;
  Serial &operator=(const Serial &) = delete;

  /***
   *  Open the serial port as long as the port name is set into the Serial object (do nothing if already open).@n
   *  If the port name is provided to the constructor then an explicit call to open is not needed.
   *  @sa Serial::Serial
   *  @throw serial::SerialIOException
   *  @throw serial::SerialInvalidArgumentException
   */
  void open();

  /***
   *  @return @p true if the port is open.
   */
  bool isOpen() const;

  /***
   *  Close the serial port.
   *  @throw serial::SerialIOException
   */
  void close();

  /***
   *  @return The number of characters available in the input buffer.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  size_t available();

  /***
   *  Block until the port is in a readable state or timeout @p read_constant_ milliseconds have elapsed.
   *  @note Not implemented on Windows.
   *  @return @p true if the port is in a readable state.
   *  @throw serial::SerialException (only on Windows systems)
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  bool waitReadable();

  /***
   *  Block until the port is in a writable state or timeout @p write_constant_ milliseconds have elapsed.
   *  @note Not implemented on Windows.
   *  @return @p true if the port is in a writable state.
   *  @throw serial::SerialException (only on Windows systems)
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  bool waitWritable();

  /***
   *  Block for a period of time corresponding to the transmission time of count characters at present serial settings.
   *  This may be used in conjunction with @p Serial::waitReadable to read larger blocks of data from the port.
   *  @sa Serial::waitReadable
   */
  void waitByteTimes(size_t count);

  /***
   *  Read at most the given amount of bytes from the serial port and store them into the given buffer.@n
   *  This method returns as soon as @p size bytes are read or when the timeout triggers (either when the inter byte or
   *  the total timeouts expire, @sa Serial::Timeout).
   *  @param buffer The array of at least the given size (the caller must size it accordingly) where the data is stored.
   *  @param size The number of bytes that should be read.
   *  @return The number of bytes actually read.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  size_t read(uint8_t *buffer, size_t size);

  /***
   *  Read at most the given amount of bytes from the serial port and store them into the given buffer.@n
   *  This method returns as soon as @p size bytes are read or when the timeout triggers (either when the inter byte or
   *  the total timeouts expire, @sa Serial::Timeout).
   *  @param buffer The vector where the data is stored.
   *  @param size The number of bytes that should be read.
   *  @return The number of bytes actually read.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  size_t read(std::vector<uint8_t> &buffer, size_t size = 1);

  /***
   *  Read at most the given amount of characters from the serial port and store them into the given buffer.@n
   *  This method returns as soon as @p size characters are read or when the timeout triggers (either when the inter
   *  byte or the total timeouts expire, @sa Serial::Timeout).
   *  @param buffer The string where the data is stored.
   *  @param size The number of characters that should be read.
   *  @return The number of characters actually read.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  size_t read(std::string &buffer, size_t size = 1);

  /***
   *  Read at most the given amount of characters from the serial port and return them as string.@n
   *  This method returns as soon as @p size characters are read or when the timeout triggers (either when the inter
   *  byte or the total timeouts expire, @sa Serial::Timeout).
   *  @param size The number of characters that should be read.
   *  @return The string where the data is stored.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  std::string read(size_t size = 1);

  /***
   *  Read characters from the serial port until the given delimiter is found and store them into the given buffer.@n
   *  This method returns as soon as the given delimiter is found or @p size characters are read or when the timeout
   *  triggers (either when the inter byte or the total timeouts expire, @sa Serial::Timeout).
   *  @param line The string where the data is stored.
   *  @param size The maximum number of characters that can be read.
   *  @param eol The string to match against for the EOL.
   *  @return The number of characters actually read.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  size_t readline(std::string &line, size_t size = 65536, const std::string &eol = "\n");

  /***
   *  Read characters from the serial port until the given delimiter is found and store them into the given buffer.@n
   *  This method returns as soon as the given delimiter is found or @p size characters are read or when the timeout
   *  triggers (either when the inter byte or the total timeouts expire, @sa Serial::Timeout).
   *  @param size The maximum number of characters that can be read.
   *  @param eol The string to match against for the EOL.
   *  @return The string where the data is stored.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  std::string readline(size_t size = 65536, const std::string &eol = "\n");

  /***
   *  Read multiple lines from the serial port (identified by the given delimiter) and store them into the given vector
   *  (one line per element of the vector).@n
   *  This method returns when @p size characters are read or when the timeout triggers (either when the inter byte or
   *  the total timeouts expire, @sa Serial::Timeout).
   *  @param size The maximum number of characters that can be read.
   *  @param eol The string to match against for the EOL.
   *  @return The vector of strings where the data lines is stored.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  std::vector<std::string> readlines(size_t size = 65536, const std::string &eol = "\n");

  /***
   *  Write at most the given amount of bytes to the serial port.@n
   *  This method returns as soon as @p size bytes are written or when the timeout triggers (either when the inter byte
   *  or the total timeouts expire, @sa Serial::Timeout).@n
   *  @param data The array of at least the given size (the caller must size it accordingly) where the data is stored.
   *  @param size The number of bytes that should be written.
   *  @return The number of bytes actually written.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  size_t write(const uint8_t *data, size_t size);

  /***
   *  Write at most the given amount of bytes to the serial port.@n
   *  This method returns as soon as @p data.size() bytes are written or when the timeout triggers (either when the
   *  inter byte or the total timeouts expire, @sa Serial::Timeout).
   *  @param data The vector where the data is stored.
   *  @return The number of bytes actually written.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  size_t write(const std::vector<uint8_t> &data);

  /***
   *  Write at most the given amount of characters to the serial port.@n
   *  This method returns as soon as @p data.size() characters are written or when the timeout triggers (either when the
   *  inter byte or the total timeouts expire, @sa Serial::Timeout).
   *  @param data The string where the data is stored.
   *  @return The number of bytes actually written.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  size_t write(const std::string &data);

  /***
   *  Set the serial port identifier.
   *  @param port_name The path of the serial port, e.g. 'COM1' on Windows and '/dev/ttyS0' on Linux.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialInvalidArgumentException
   */
  void setPort(const std::string &port_name);

  /***
   *  @return The serial port identifier.
   */
  std::string getPort() const;

  /***
   *  Sets the timeout for reads and writes using the Timeout struct.
   *  @param timeout A Timeout struct containing the inter byte timeout, and read/write timeout constants and multipliers.
   *  @sa Serial::Timeout
   *  @throw serial::SerialIOException (only on Windows systems)
   *  @throw serial::SerialInvalidArgumentException (only on Windows systems)
   */
  void setTimeout(const Timeout &timeout);

  /***
   *  Sets the timeouts for reads and writes.
   *  @param inter_byte The maximum time allowed to elapse before the arrival of the next byte.
   *  @param read_constant A constant used to calculate the total timeout period for read operations.
   *  @param read_multiplier The multiplier used to calculate the total timeout period for read operations.
   *  @param write_constant A constant used to calculate the total timeout period for write operations.
   *  @param write_multiplier The multiplier used to calculate the total timeout period for write operations.
   *  @sa Serial::Timeout
   *  @throw serial::SerialIOException (only on Windows systems)
   *  @throw serial::SerialInvalidArgumentException (only on Windows systems)
   */
  void setTimeout(uint32_t inter_byte, uint32_t read_constant, uint32_t read_multiplier, uint32_t write_constant, uint32_t write_multiplier);

  /***
   *  @return A Timeout struct containing the inter byte timeout, and read/write timeout constants and multipliers.
   *  @sa Serial::Timeout
   */
  Timeout getTimeout() const;

  /***
   *  Sets the baudrate for the serial port.
   *  @param baudrate The baud rate of the serial communication.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialInvalidArgumentException
   */
  void setBaudrate(uint32_t baudrate);

  /***
   *  @return The baud rate of the serial communication.
   */
  uint32_t getBaudrate() const;

  /***
   *  Sets the byte size for the serial port.
   *  @param bytesize The payload size  of the serial communication.
   *  @sa bytesize_t
   *  @throw serial::SerialIOException
   *  @throw serial::SerialInvalidArgumentException
   */
  void setBytesize(bytesize_t bytesize);

  /***
   *  @return The payload size  of the serial communication.
   *  @sa bytesize_t
   */
  bytesize_t getBytesize() const;

  /***
   *  Sets the parity for the serial port.
   *  @param parity The parity method of the serial communication.
   *  @sa parity_t
   *  @throw serial::SerialIOException
   *  @throw serial::SerialInvalidArgumentException
   */
  void setParity(parity_t parity);

  /***
   *  @return The parity method of the serial communication.
   *  @sa parity_t
   */
  parity_t getParity() const;

  /***
   *  Sets the stop bits for the serial port.
   *  @param stopbits The number of stop bits of the serial communication.
   *  @sa stopbits_t
   *  @throw serial::SerialIOException
   *  @throw serial::SerialInvalidArgumentException
   */
  void setStopbits(stopbits_t stopbits);

  /***
   *  @return The number of stop bits of the serial communication.
   *  @sa stopbits_t
   */
  stopbits_t getStopbits() const;

  /***
   *  Sets the flow control for the serial port.
   *  @param flowcontrol The type of flow control of the serial communication.
   *  @sa flowcontrol_t
   *  @throw serial::SerialIOException
   *  @throw serial::SerialInvalidArgumentException
   */
  void setFlowcontrol(flowcontrol_t flowcontrol);

  /***
   *  @return The type of flow control of the serial communication.
   *  @sa flowcontrol_t
   */
  flowcontrol_t getFlowcontrol() const;

  /***
   *  Flush the input and output buffers of the serial communication.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  void flush();

  /***
   *  Flush the input buffer of the serial communication.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  void flushInput();

  /***
   *  Flush the output buffer of the serial communication.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  void flushOutput();

  /***
   *  Sends the serial break signal, see @p tcsendbreak().
   *  @note Not implemented on Windows.
   *  @throw serial::SerialException (only on Windows systems)
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  void sendBreak(int duration);

  /***
   *  Set the break condition to the given level (default is @p true, high).
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  void setBreak(bool level = true);

  /***
   *  Set the RTS handshaking line to the given level (default is @p true, high).
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  void setRTS(bool level = true);

  /***
   *  Set the DTR handshaking line to the given level (default is @p true, high).
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  void setDTR(bool level = true);

  /***
   *  Blocks until CTS, DSR, RI, CD changes or something interrupts it.
   *  @return @p true if one of the lines changed, @p false otherwise.
   *  @throw serial::SerialException (#ifndef TIOCMIWAIT on unix systems)
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  void waitForModemChanges();

  /***
   *  @return The current status of the CTS line.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  bool getCTS();

  /***
   *  @return The current status of the DSR line.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  bool getDSR();

  /***
   *  @return The current status of the RI line.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  bool getRI();

  /***
   *  @return The current status of the CD line.
   *  @throw serial::SerialIOException
   *  @throw serial::SerialPortNotOpenException
   */
  bool getCD();

 private:
  class SerialImpl;
  std::unique_ptr<SerialImpl> pimpl_;
  std::mutex read_mutex_;
  std::mutex write_mutex_;

  size_t readline_(std::string &line, size_t size = 65536, const std::string &eol = "\n");  // core method which does not lock on mutex
};

class SerialException : public std::runtime_error {
 public:
  SerialException() : SerialException("generic fault") {}
  explicit SerialException(const std::string &what_arg) : std::runtime_error("Serial Exception: " + what_arg + ".") {}
};

class SerialInvalidArgumentException : public std::invalid_argument {
 public:
  SerialInvalidArgumentException() : SerialInvalidArgumentException("generic fault") {}
  explicit SerialInvalidArgumentException(const std::string &what_arg) : std::invalid_argument("Serial Invalid Argument Exception: " + what_arg + ".") {}
};

class SerialIOException : public std::runtime_error {
 public:
  SerialIOException() : SerialIOException("generic fault") {}
  explicit SerialIOException(const std::string &what_arg) : std::runtime_error("Serial IO Exception: " + what_arg + "'.") {}
  explicit SerialIOException(const std::string &what_arg, uint32_t error) : std::runtime_error("Serial IO Exception: " + what_arg + ", error has been set to '" + std::to_string(error) + "'.") {}
};

class SerialPortNotOpenException : public std::runtime_error {
 public:
  SerialPortNotOpenException() : std::runtime_error("Serial Port Not Open Exception.") {}
};

/***
 *  Structure that describes a serial device.
 */
class PortInfo {
 public:
  PortInfo() = default;
  ~PortInfo() = default;

  uint16_t busnum {0};
  uint16_t devnum {0};
  uint16_t id_product {0};
  uint16_t id_vendor {0};
  std::string manufacturer;
  std::string product;
  std::string serial_number;
  std::string serial_port;

  /***
   *  Fill all the @p PortInfo members with the data from the given serial port node (if found in the system).@n
   *  If the serial port is not a USB device, only the @p serial_port field is filled.
   *  @param serial_port_name The serial port name to search for, e.g. 'COM1' on Windows and '/dev/ttyS0' on Linux.
   *  @return @p 0 on success, @p -1 otherwise.
   */
  int getPortInfo(const std::string &serial_port_name);
};

/***
 *  Return through the given vector the @p PortInfo of all the serial port connected to the system.
 *  @param serial_ports The vector of serial port info.
 *  @return @p 0 on success, @p -1 otherwise.
 *  @sa PortInfo
 */
int getPortsInfo(std::vector<PortInfo> &serial_ports);

/***
 *  Return through the given vector the names of all the serial port connected to the system.
 *  @param serial_port_names The vector of serial port names.
 *  @return @p 0 on success, @p -1 otherwise.
 */
int getPortsList(std::vector<std::string> &serial_port_names);

} // namespace serial

#endif
