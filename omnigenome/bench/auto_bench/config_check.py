








one_shot_messages = set()


def config_check(args):
    
    try:
        if "use_amp" in args:
            assert args["use_amp"] in {True, False}
        if "patience" in args:
            assert args["patience"] > 0

    except AssertionError as e:
        raise RuntimeError(
            "Exception: {}. Some parameters are not valid, please see the main example.".format(
                e
            )
        )
