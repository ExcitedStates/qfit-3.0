"""Contains custom argparse actions & formatters."""

import argparse
from pathlib import Path


class ToggleActionFlag(argparse.Action):
    """Adds a 'no' prefix to disable store_true actions.

    For example, --debug will have an additional --no-debug to explicitly disable it.
    """

    def __init__(self, option_strings, dest=None, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

        if len(option_strings) != 1:
            raise argparse.ArgumentError(
                self,
                f"An argument of type {self.__class__.__name__} "
                f"can only have one opt-string.",
            )

        option_string = option_strings[0].lstrip("-")
        self.option_strings = ["--" + option_string, "--no-" + option_string]
        self.dest = option_string.replace("-", "_") if dest is None else dest
        self.nargs = 0
        self.const = None

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string.startswith("--no-"):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)


class ToggleActionFlagFormatter(argparse.HelpFormatter):
    """Condenses --help output.

    This changes the --help output, what is originally this:
        --file, --no-file, -f
    will be condensed like this:
        --[no-]file, -f
    """

    # ht: https://stackoverflow.com/questions/9234258/in-python-argparse-is-it-possible-to-have-paired-no-something-something-arg/36803315#36803315

    def _format_action_invocation(self, action):
        if isinstance(action, ToggleActionFlag):
            return ", ".join(
                [action.option_strings[0][:2] + "[no-]" + action.option_strings[0][2:]]
                + action.option_strings[2:]
            )
        else:
            return super()._format_action_invocation(action)


class CustomHelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
    ToggleActionFlagFormatter,
):
    pass


class ValidateMapFileArgument(argparse.Action):
    """Checks that a valid map file was provided."""

    extension_choices = set(
        (
            ".ccp4",
            ".mtz",
            ".mrc",
            ".map",
        )
    )

    def __call__(self, parser, namespace, value, option_string=None):
        fname = Path(value)
        msg = ""

        if fname.suffix not in self.extension_choices:
            msg += f"Provided map ({value}) is not a supported filetype: {self.extension_choices}."
            if fname.suffix in ValidateStructureFileArgument.extension_choices:
                msg += "\nDid you get the argument order (map structure) right?"
            parser.error(msg)

        if not fname.is_file():
            parser.error(f"Could not find map file ({value}).")

        setattr(namespace, self.dest, value)


class ValidateStructureFileArgument(argparse.Action):
    """Checks that a valid structure file was provided."""

    extension_choices = set((".pdb",))

    def __call__(self, parser, namespace, value, option_string=None):
        fname = Path(value)
        msg = ""

        if fname.suffix not in self.extension_choices:
            msg += f"Provided structure ({value}) is not a supported filetype: {self.extension_choices}."
            if fname.suffix in ValidateMapFileArgument.extension_choices:
                msg += "\nDid you get the argument order (map structure) right?"
            parser.error(msg)

        if not fname.is_file():
            parser.error(f"Could not find structure file ({value}).")

        setattr(namespace, self.dest, value)
