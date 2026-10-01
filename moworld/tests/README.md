# Retained Runtime Tests

This directory retains upstream unit tests for shared runtime components, including data processing, autoencoders, text encoders, Wan scheduling, distributed execution, checkpoint handling, and optimizers.

Other-model-specific tests and upstream system-test launchers have been removed. These remaining unit tests do not constitute end-to-end MoWorld validation; some require Ascend devices and distributed execution.

Prepare the environment described in the [usage guide](../../USAGE.md#installation) before selecting tests for the component being changed. MoWorld pretraining and data-preparation launchers are in [examples/moworld](../examples/moworld/).
