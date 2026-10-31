"""Lightweight CLI options shared by the launcher and standalone service."""


def add_runtime_arguments(parser, *, default_device="cpu"):
    parser.add_argument("--device", default=default_device,
                        help="shared device for Gloo mode / reference device, e.g. cpu or cuda:0")
    parser.add_argument("--stage-devices", nargs=3, metavar=("V", "P", "D"),
                        help="three logical devices after CUDA_VISIBLE_DEVICES remapping")
    parser.add_argument("--transport", choices=("gloo", "nccl"), default="gloo",
                        help="tensor backend; NCCL defaults to cuda:0 cuda:1 cuda:2")
    parser.add_argument("--no-handoff-checksum", action="store_true",
                        help="skip CPU SHA256 payload mirrors; metadata/order checks remain enabled")
