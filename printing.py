from torch import Tensor, nn

from torch.utils.data import DataLoader


from constants import NnStage


def print_datashape(dataloader: DataLoader, nn_stage: NnStage) -> None:
    X: Tensor
    Y: Tensor
    for X, Y in dataloader:
        print(f"Shape of {nn_stage.value} input data: {X.shape}")
        print(f"Shape of {nn_stage.value} output data: {Y.shape}")
        return


def print_model(model: nn.Module) -> None:
    print("--------------------")
    print("This neural network model:")
    print(model)
    print("--------------------")


def print_test_results(correct: float, test_loss: float) -> None:
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def print_done(epoch: int, accuracy: float) -> None:
    print(f"Done! It took {epoch} epochs to obtain an accuracy of {accuracy}")


def print_model_found(model_str: str) -> None:
    print(f"Found the following model: {model_str}.\n" f"This model will be loaded in.")
