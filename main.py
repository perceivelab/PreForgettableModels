from cli import parse_args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
