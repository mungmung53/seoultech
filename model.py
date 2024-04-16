{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMj4QZC5+nMV0Zxh9+u28GX",
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
        "<a href=\"https://colab.research.google.com/github/mungmung53/seoultech/blob/main/model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch.nn as nn\n",
        "\n",
        "class LeNet5(nn.Module):\n",
        "    \"\"\" LeNet-5 (LeCun et al., 1998)\n",
        "\n",
        "        - For a detailed architecture, refer to the lecture note\n",
        "        - Freely choose activation functions as you want\n",
        "        - For subsampling, use max pooling with kernel_size = (2,2)\n",
        "        - Output should be a logit vector\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(self):\n",
        "        super(LeNet5, self).__init__()\n",
        "        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel size\n",
        "        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input channels, 16 output channels, 5x5 kernel size\n",
        "        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Fully connected layer\n",
        "        self.fc2 = nn.Linear(120, 84)  # Fully connected layer\n",
        "        self.fc3 = nn.Linear(84, 10)  # Output layer\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = nn.functional.relu(self.conv1(x))\n",
        "        x = nn.functional.max_pool2d(x, 2)\n",
        "        x = nn.functional.relu(self.conv2(x))\n",
        "        x = nn.functional.max_pool2d(x, 2)\n",
        "        x = x.view(-1, 16 * 4 * 4)  # Flatten the tensor\n",
        "        x = nn.functional.relu(self.fc1(x))\n",
        "        x = nn.functional.relu(self.fc2(x))\n",
        "        x = self.fc3(x)\n",
        "        return x\n",
        "\n",
        "class CustomMLP(nn.Module):\n",
        "    \"\"\" Your custom MLP model\n",
        "\n",
        "        - Note that the number of model parameters should be about the same\n",
        "          with LeNet-5\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(self):\n",
        "        super(CustomMLP, self).__init__()\n",
        "        self.fc1 = nn.Linear(28 * 28, 500)\n",
        "        self.fc2 = nn.Linear(500, 200)\n",
        "        self.fc3 = nn.Linear(200, 10)\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = x.view(-1, 28 * 28)\n",
        "        x = nn.functional.relu(self.fc1(x))\n",
        "        x = nn.functional.relu(self.fc2(x))\n",
        "        x = self.fc3(x)\n",
        "        return x\n"
      ],
      "metadata": {
        "id": "a_DL9H6cKNCg"
      },
      "execution_count": 1,
      "outputs": []
    }
  ]
}