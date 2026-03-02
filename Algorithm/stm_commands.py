import re


def to_stm_commands(commands, turn_unit_deg: float | None = None):
    """Translate planner commands to STM command strings (newline-terminated)."""
    turn_map = {
        "FR": "TR",
        "FL": "TL",
        "BR": "BTR",
        "BL": "BTL",
    }
    turn_with_angle_pattern = re.compile(r"^(FR|FL|BR|BL)(\d{1,3})$")

    stm_commands = []
    if turn_unit_deg is not None and turn_unit_deg <= 0:
        raise ValueError("turn_unit_deg must be positive when provided")

    for command in commands:
        if command.startswith("FW"):
            distance = int(command[2:])
            stm_commands.append(f"SF{distance:03d}\n")
            continue

        if command.startswith("BW"):
            distance = int(command[2:])
            stm_commands.append(f"SB{distance:03d}\n")
            continue

        turn_match = turn_with_angle_pattern.fullmatch(command)
        if turn_match:
            prefix = turn_match.group(1)
            angle_deg = int(turn_match.group(2))
            stm_commands.append(f"{turn_map[prefix]}{angle_deg:03d}\n")
            continue

        prefix = command[:2]
        if prefix in turn_map:
            angle_deg = 0
            if turn_unit_deg is not None:
                angle_deg = max(1, int(round(turn_unit_deg)))
            stm_commands.append(f"{turn_map[prefix]}{angle_deg:03d}\n")
            continue

        if command.startswith("SNAP"):
            stm_commands.append("PING20\n")
            stm_commands.append(f"{command}\n")
            continue

        if command == "FIN":
            stm_commands.append("FIN\n")
            continue

        if command == "STOP":
            stm_commands.append("STOP\n")
            continue

        raise ValueError(f"Unsupported planner command: {command}")

    return stm_commands
