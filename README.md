# Valohai Pytorch Lightning Example

This repository serves as an example for the [Valohai MLOps platform][vh]. It implements handwritten digit detection
using [Pytorch Lightning][pl].

[vh]: https://valohai.com/
[pl]: https://lightning.ai/
[app]: https://app.valohai.com

We want to demonstrate how you can use Valohai with PyTorch Lightning.
This repository contains an example that shows how to log data to Valohai using PyTorch Lightning hooks.

Let's walk through a few steps to set up the project.

## <div align="center">Installation</div>

Login to the [Valohai app][app] and create a new project.

### Configure the repository:

<details open>
<summary>Using UI</summary>

Configure this repository as the project's repository, by following these steps:

1. Go to your project's page.
2. Navigate to the Settings tab.
3. Under the Repository section, locate the URL field.
4. Enter the URL of this repository.
5. Click on the Save button to save the changes.
</details>

<details open>
<summary>Using terminal</summary>

To run the code on Valohai using the terminal, follow these steps:

1. Install Valohai on your machine by running the following command:

```bash
pip install valohai-cli valohai-utils
```

2. Log in to Valohai from the terminal using the command:

```bash
vh login
```

3. Create a project for your Valohai workflow.
   Start by creating a directory for your project:

```bash
mkdir valohai-pytorch-lightning-example
cd valohai-pytorch-lightning-example
```

Then, create the Valohai project:

```bash
vh project create
```

4. Clone the repository to your local machine:

```bash
git clone https://github.com/valohai/pytorch-lightning-example.git .
```
</details>

## <div align="center">Running Execution</div>

<details open>
<summary>Using UI</summary>

1. Go to the Executions tab in your project.
2. Create a new execution by selecting the step - _train-mnist_
3. Customize the execution parameters if needed.
4. Start the execution to run the selected step.

</details>

<details open>
<summary>Using terminal</summary>

To run the step, execute the following command:

```bash
vh execution run train-mnist --adhoc
```
</details>

## <div align="center">Pytorch Lightning Hooks</div>

To collect metadata and log it to Valohai, we can utilize the `on_train_epoch_end` hook provided by PyTorch Lightning. 
Within this hook, we can access the `valohai_utils` module, which offers a convenient way to interact with the Valohai platform and log metadata.

    def on_train_epoch_end(self):
        with valohai.metadata.logger() as logger:
            train_loss = ...
            train_acc = ...
            logger.log("epoch", self.current_epoch + 1)
            logger.log("train_acc", train_acc)
            logger.log("train_loss", train_loss)

In the above code, you can replace `train_loss` and `train_acc` with the actual values you want to log. The `logger.log` function allows you to log different metrics or values by specifying the key and the corresponding value.

By utilizing this hook and the `valohai_utils` module, you can easily log the desired metadata to Valohai during the training process.

## <div align="center">Save best model</div>

The `save_best_model` method performs the following steps:

1. Load the best checkpoint path. 
2. Load the model from the checkpoint. 
3. Save the best model's state dictionary to a designated output path. 
4. Create a metadata JSON file for the best model, including a Valohai alias.

This method saves the best model obtained during training, allowing you to easily save the best model in a format suitable for deployment.

