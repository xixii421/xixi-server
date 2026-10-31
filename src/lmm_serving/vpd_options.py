"""Lightweight CLI options shared by the launcher and standalone service."""


def add_placement_arguments(parser):
    parser.add_argument("--stage-devices", nargs=3, metavar=("V", "P", "D"),
                        help="three logical devices after CUDA_VISIBLE_DEVICES remapping")
    parser.add_argument("--vision-devices", nargs="+", help="vision worker device pool")
    parser.add_argument("--prefill-devices", nargs="+", help="prefill worker device pool")
    parser.add_argument("--decode-devices", nargs="+", help="decode worker device pool")


def add_runtime_arguments(parser, *, default_device="cpu"):
    parser.add_argument("--device", default=default_device,
                        help="shared device for Gloo mode / reference device, e.g. cpu or cuda:0")
    add_placement_arguments(parser)
    parser.add_argument("--transport", choices=("gloo", "nccl"), default="gloo",
                        help="tensor backend; NCCL defaults to cuda:0 cuda:1 cuda:2")
    parser.add_argument("--no-handoff-checksum", action="store_true",
                        help="skip CPU SHA256 payload mirrors; metadata/order checks remain enabled")


def placement_kwargs(args):
    """三阶段单设备旧参数与三组设备池参数只选一种。"""
    pools=(args.vision_devices,args.prefill_devices,args.decode_devices)
    if any(pool is not None for pool in pools):
        if args.stage_devices is not None or not all(pool is not None for pool in pools):
            raise ValueError("provide all three device pools, without --stage-devices")
        return {"stage_devices":None,"stage_pools":tuple(tuple(pool) for pool in pools)}
    return {"stage_devices":tuple(args.stage_devices) if args.stage_devices else None,
            "stage_pools":None}
