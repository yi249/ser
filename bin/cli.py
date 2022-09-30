import typer
from ser.infer import infers
main = typer.Typer()


@main.command()
def train():
    print("This is where the training code will go")

@main.command()
def infer():
    infers()