import socket
import time
from pymodbus.client import ModbusTcpClient
from flipfocus2 import run_edge_detection

# =========================
# CONFIG
# =========================
HOST = "0.0.0.0"
PORT = 29999
TRIGGER_MESSAGE = "run_my_script"

ROBOT_IP = "10.241.34.45"
MODBUS_PORT = 502
REGISTER_ADDRESS = 129

WAITING_VALUE = 0
PASS_VALUE = 3
FAIL_VALUE = 4

# How long to leave the 3 or 4 on the register before resetting to 0
RESULT_HOLD_TIME = 1


def convert_vision_result_to_shift_value(result):
    """
    Convert the vision result dictionary into the robot decision value.
    """
    if result["passed"]:
        return PASS_VALUE
    return FAIL_VALUE


def update_shift_register(shift_value):
    """
    Connect to the robot over Modbus TCP and write the shift value
    into the chosen register.
    """
    client = ModbusTcpClient(ROBOT_IP, port=MODBUS_PORT)

    try:
        if not client.connect():
            print("Connection to Rosie failed.")
            return False

        response = client.write_register(REGISTER_ADDRESS, shift_value)

        if response.isError():
            print(
                f"Failed to write {shift_value} to register {REGISTER_ADDRESS}."
            )
            return False

        print(
            f"Successfully sent {shift_value} to MODBUS register {REGISTER_ADDRESS}."
        )
        return True

    except Exception as e:
        print(f"Modbus write error: {e}")
        return False

    finally:
        client.close()


def my_custom_logic():
    """
    Main workflow after the robot sends the trigger:
    1. Reset register to 0
    2. Run edge detection
    3. Convert result into a robot decision value
    4. Write that value to the robot register
    5. Hold briefly so robot can read it
    6. Reset register back to 0
    """
    print("Robot triggered the script.")

    # Reset to 0 before starting
    print("Resetting shift register to 0 before vision run...")
    reset_ok = update_shift_register(WAITING_VALUE)
    if not reset_ok:
        print("Warning: could not reset register to 0 before vision run.")

    # Run vision
    print("Running vision...")
    result = run_edge_detection()

    print(f"Status: {result['status']}")
    print(f"Passed: {result['passed']}")

    # Convert result to robot value
    shift_value = convert_vision_result_to_shift_value(result)
    print(f"Shift value to send: {shift_value}")

    # Send 3 or 4
    write_ok = update_shift_register(shift_value)
    if not write_ok:
        print("Warning: failed to write final pass/fail value.")
        return result

    print("Vision finished and shift register updated.")

    # Hold the result briefly so the robot has time to read it
    time.sleep(RESULT_HOLD_TIME)

    # Reset back to 0 after sending result
    print("Resetting shift register back to 0 after sending result...")
    reset_after_ok = update_shift_register(WAITING_VALUE)
    if not reset_after_ok:
        print("Warning: failed to reset register to 0 after final result.")
    else:
        print("Shift register reset to 0.")

    return result


def handle_connection(conn, addr):
    """
    Handle one robot connection.
    """
    print(f"Connected by {addr}")

    try:
        data = conn.recv(1024).decode("utf-8").strip()
        print(f"Received: {data}")

        if data == TRIGGER_MESSAGE:
            try:
                my_custom_logic()
            except Exception as e:
                print(f"Vision logic error: {e}")
        else:
            print("Trigger message did not match.")

    except Exception as e:
        print(f"Socket handling error: {e}")

    finally:
        print("Connection closed.\n")


def start_server():
    """
    Start a TCP server that waits for the robot to connect and send
    the trigger message.
    This keeps running forever until you stop the Python script.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)

        print(f"Server started. Waiting for UR5 on {HOST}:{PORT}...")

        while True:
            try:
                conn, addr = s.accept()
                with conn:
                    handle_connection(conn, addr)
            except Exception as e:
                print(f"Server loop error: {e}")


if __name__ == "__main__":
    start_server()
