{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1qL51RQ2pJHuM4y4NoNhvEwan_leuHyJB",
      "authorship_tag": "ABX9TyNt+AERNoZFhepuIOvmbE9x",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/mungmung53/seoultech/blob/main/main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import DataLoader\n",
        "from torchvision import transforms\n",
        "from dataset import MNIST\n",
        "from model import LeNet5, CustomMLP\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Set device (GPU or CPU)\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "\n",
        "# Hyperparameters\n",
        "batch_size = 64\n",
        "learning_rate = 0.01\n",
        "momentum = 0.9\n",
        "num_epochs = 10\n",
        "\n",
        "# Transformations for preprocessing\n",
        "transform = transforms.Compose([\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize((0.1307,), (0.3081,))\n",
        "])\n",
        "\n",
        "def main():\n",
        "    # Load dataset\n",
        "    train_dataset = MNIST(data_dir='/content/mnist_data/mnist-classification/data/train.tar', transform=transform)\n",
        "    test_dataset = MNIST(data_dir='/content/mnist_data/mnist-classification/data/test.tar', transform=transform)\n",
        "\n",
        "    # Create data loaders\n",
        "    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)\n",
        "    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)\n",
        "\n",
        "    # Initialize models\n",
        "    lenet5_model = LeNet5().to(device)\n",
        "    custom_mlp_model = CustomMLP().to(device)\n",
        "\n",
        "    # Define loss function and optimizer\n",
        "    criterion = nn.CrossEntropyLoss()\n",
        "    lenet5_optimizer = optim.SGD(lenet5_model.parameters(), lr=learning_rate, momentum=momentum)\n",
        "    custom_mlp_optimizer = optim.SGD(custom_mlp_model.parameters(), lr=learning_rate, momentum=momentum)\n",
        "\n",
        "    # Lists to store training progress for plotting\n",
        "    lenet5_train_loss = []\n",
        "    lenet5_train_acc = []\n",
        "    custom_mlp_train_loss = []\n",
        "    custom_mlp_train_acc = []\n",
        "    lenet5_test_loss = []\n",
        "    lenet5_test_acc = []\n",
        "    custom_mlp_test_loss = []\n",
        "    custom_mlp_test_acc = []\n",
        "\n",
        "    # Training loop\n",
        "    for epoch in range(num_epochs):\n",
        "        lenet5_model.train()  # Set model to training mode\n",
        "        custom_mlp_model.train()\n",
        "\n",
        "        lenet5_epoch_loss, lenet5_epoch_acc = train(lenet5_model, train_loader, device, criterion, lenet5_optimizer)\n",
        "        custom_mlp_epoch_loss, custom_mlp_epoch_acc = train(custom_mlp_model, train_loader, device, criterion, custom_mlp_optimizer)\n",
        "\n",
        "        lenet5_train_loss.append(lenet5_epoch_loss)\n",
        "        lenet5_train_acc.append(lenet5_epoch_acc)\n",
        "        custom_mlp_train_loss.append(custom_mlp_epoch_loss)\n",
        "        custom_mlp_train_acc.append(custom_mlp_epoch_acc)\n",
        "\n",
        "        # Evaluate on test set\n",
        "        lenet5_model.eval()  # Set model to evaluation mode\n",
        "        custom_mlp_model.eval()\n",
        "\n",
        "        lenet5_test_epoch_loss, lenet5_test_epoch_acc = test(lenet5_model, test_loader, device, criterion)\n",
        "        custom_mlp_test_epoch_loss, custom_mlp_test_epoch_acc = test(custom_mlp_model, test_loader, device, criterion)\n",
        "\n",
        "        lenet5_test_loss.append(lenet5_test_epoch_loss)\n",
        "        lenet5_test_acc.append(lenet5_test_epoch_acc)\n",
        "        custom_mlp_test_loss.append(custom_mlp_test_epoch_loss)\n",
        "        custom_mlp_test_acc.append(custom_mlp_test_epoch_acc)\n",
        "\n",
        "        print(f\"Epoch [{epoch + 1}/{num_epochs}], \"\n",
        "              f\"LeNet-5 Train Loss: {lenet5_epoch_loss:.4f}, LeNet-5 Train Acc: {lenet5_epoch_acc:.4f}, \"\n",
        "              f\"Custom MLP Train Loss: {custom_mlp_epoch_loss:.4f}, Custom MLP Train Acc: {custom_mlp_epoch_acc:.4f}, \"\n",
        "              f\"LeNet-5 Test Loss: {lenet5_test_epoch_loss:.4f}, LeNet-5 Test Acc: {lenet5_test_epoch_acc:.4f}, \"\n",
        "              f\"Custom MLP Test Loss: {custom_mlp_test_epoch_loss:.4f}, Custom MLP Test Acc: {custom_mlp_test_epoch_acc:.4f}\")\n",
        "\n",
        "    # Plotting\n",
        "    plt.figure(figsize=(10, 5))\n",
        "    plt.subplot(1, 2, 1)\n",
        "    plt.plot(lenet5_train_loss, label='LeNet-5 Train Loss')\n",
        "    plt.plot(custom_mlp_train_loss, label='Custom MLP Train Loss')\n",
        "    plt.plot(lenet5_test_loss, label='LeNet-5 Test Loss')\n",
        "    plt.plot(custom_mlp_test_loss, label='Custom MLP Test Loss')\n",
        "    plt.xlabel('Epochs')\n",
        "    plt.ylabel('Loss')\n",
        "    plt.legend()\n",
        "\n",
        "    plt.subplot(1, 2, 2)\n",
        "    plt.plot(lenet5_train_acc, label='LeNet-5 Train Acc')\n",
        "    plt.plot(custom_mlp_train_acc, label='Custom MLP Train Acc')\n",
        "    plt.plot(lenet5_test_acc, label='LeNet-5 Test Acc')\n",
        "    plt.plot(custom_mlp_test_acc, label='Custom MLP Test Acc')\n",
        "    plt.xlabel('Epochs')\n",
        "    plt.ylabel('Accuracy')\n",
        "    plt.legend()\n",
        "\n",
        "    plt.show()\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    main()\n"
      ],
      "metadata": {
        "id": "WybJYtl2jUi3",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 312
        },
        "outputId": "e8619c64-9c18-45d2-b0f8-3f5a95af5e20"
      },
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "error",
          "ename": "TypeError",
          "evalue": "MNIST.__init__() got an unexpected keyword argument 'transform'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-64-d9b0ebdd59c8>\u001b[0m in \u001b[0;36m<cell line: 106>\u001b[0;34m()\u001b[0m\n\u001b[1;32m    105\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    106\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0m__name__\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'__main__'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 107\u001b[0;31m     \u001b[0mmain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m<ipython-input-64-d9b0ebdd59c8>\u001b[0m in \u001b[0;36mmain\u001b[0;34m()\u001b[0m\n\u001b[1;32m     25\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0mmain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     26\u001b[0m     \u001b[0;31m# Load dataset\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 27\u001b[0;31m     \u001b[0mtrain_dataset\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mMNIST\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata_dir\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'/content/mnist_data/mnist-classification/data/train.tar'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtransform\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtransform\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     28\u001b[0m     \u001b[0mtest_dataset\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mMNIST\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata_dir\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'/content/mnist_data/mnist-classification/data/test.tar'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtransform\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtransform\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     29\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mTypeError\u001b[0m: MNIST.__init__() got an unexpected keyword argument 'transform'"
          ]
        }
      ]
    }
  ]
}