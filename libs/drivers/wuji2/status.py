import argparse

import wuji_sdk
from wuji_sdk import SdkManager


def _scan_hand2(manager):
    devs = manager.scan()
    cands = [d for d in devs if d.sn.startswith("WH2")]
    if not cands:
        raise SystemExit("No Wuji Hand 2 discovered (no SN starts with 'WH2').")
    return manager.connect(sn=cands[0].sn, device_name="wuji_hand_2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip", default="", help="optional override; auto-scan when empty"
    )
    args = parser.parse_args()

    wuji_sdk.set_log_level("warn")
    manager = SdkManager.instance()
    hand = (
        manager.connect(address=args.ip, device_name="wuji_hand_2")
        if args.ip
        else _scan_hand2(manager)
    )
    try:
        diags = hand.diagnostics().get()
        for joint, d in zip(hand.joints(), diags, strict=True):
            if d is not None:
                print(
                    f"  {joint.label:<12} pos={d.position:>8.3f}  vel={d.velocity:>8.3f}  "
                    f"temp={d.temperature:.1f}C  vbus={d.vbus:.1f}V  "
                    f"state={d.inverter_state}  fault={d.fault_code}"
                )
            else:
                print(f"  {joint.label:<12} OFFLINE")
        online = hand.online_joints_count().get()
        print(f"\n{online}/20 joints online")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        manager.disconnect_all()


if __name__ == "__main__":
    main()
