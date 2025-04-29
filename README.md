# MNIST Handwritten Digit Classification (PyTorch, ANN)

Classify handwritten digits (0–9) from the MNIST dataset using a simple Artificial Neural Network (ANN) built with PyTorch.

---

## Features

- Loads + normalizes MNIST data
- Visualizes random samples
- Custom ANN with two hidden layers
- Training with loss plot
- Evaluation with test accuracy
- Works on CPU or GPU

---

## Quickstart

<details>
<summary><strong>Setup & Run</strong></summary>

1. **Clone & Enter Project**
    ```sh
    git clone https://github.com/YOURUSER/YOURREPONAME.git
    cd YOURREPONAME
    ```

2. **(Optional) Create virtual environment**
    ```sh
    python -m venv venv
    # Activate (Windows)
    venv\Scripts\activate
    # Activate (macOS/Linux)
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```sh
    pip install torch torchvision matplotlib
    ```

4. **Run the script**
    ```sh
    python mnist_ann.py
    ```

</details>

---

## Visual Studio Code (VSCode)

- Use [VSCode](https://code.visualstudio.com/) for Python development
- Install the **Python** extension
- Use the built-in terminal (Ctrl+Shift+`) to activate your environment and run scripts
- Edit/debug your code with syntax highlighting, linting, and breakpoints

---

## How it Works

- **Data:** Downloads and normalizes MNIST digits.
- **Visualization:** Shows a few random labels/images.
- **Model:** Simple ANN—input layer → 128 (ReLU) → 64 (ReLU) → 10 outputs.
- **Training:** Uses Adam optimizer & cross-entropy loss, plotted per epoch.
- **Testing:** Prints classification accuracy on the test set.

---

## Troubleshooting

- *CUDA not used?* Install the [correct PyTorch for CUDA](https://pytorch.org/get-started/locally/).
- *Plots not showing?* Run the script directly (not line by line).
- *PIL errors?* Run: `pip install pillow`

---

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [MNIST Info](http://yann.lecun.com/exdb/mnist/)
- [VSCode Python](https://code.visualstudio.com/docs/python/python-tutorial)

---

## License

MIT or your choice.
