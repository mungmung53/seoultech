{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1Pp_dLu76dmM-WDWmjLe-hGdr6HKZxnrU",
      "authorship_tag": "ABX9TyOIpyKSY8dSlmmBfZRVLG3e",
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
        "<a href=\"https://colab.research.google.com/github/mungmung53/seoultech/blob/main/dataset.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "\n",
        "# 마운트된 Google 드라이브의 경로 출력\n",
        "print(\"Google 드라이브 경로:\", os.getcwd())\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "s0zS_8IQFp5l",
        "outputId": "d9c7c735-bcd8-465b-a028-a46b82e2ae49"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Google 드라이브 경로: /content\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import shutil\n",
        "\n",
        "# 압축 파일 경로\n",
        "zip_file_path = \"/content/drive/MyDrive/SeoulTech/인공신경망과 딥러닝/mnist-classification.zip\"\n",
        "# 압축을 해제할 디렉토리 경로\n",
        "extracted_dir_path = \"/content/mnist_data\"\n",
        "\n",
        "# 압축 해제\n",
        "shutil.unpack_archive(zip_file_path, extracted_dir_path)\n",
        "\n",
        "# 압축 해제된 데이터 디렉토리 내 파일 목록 확인\n",
        "print(\"압축 해제된 데이터 디렉토리 내 파일 목록:\", os.listdir(extracted_dir_path))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aYv4km7UFZ2v",
        "outputId": "139b75ac-c80b-44e5-f954-2678bfa1c49b"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "압축 해제된 데이터 디렉토리 내 파일 목록: ['mnist-classification']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os, io, tarfile\n",
        "from PIL import Image\n",
        "from torch.utils.data import Dataset\n",
        "from torchvision import transforms\n",
        "\n",
        "class MNIST(Dataset):\n",
        "    \"\"\" MNIST dataset\n",
        "    Args:\n",
        "        data_dir: directory path containing tar files\n",
        "    Note:\n",
        "        Each image should be preprocessed as follows:\n",
        "            - First, all values should be in a range of [0,1]\n",
        "            - Subtract mean of 0.1307, and divide by std 0.3081\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(self, data_dir):\n",
        "        self.data_dir = data_dir\n",
        "        self.tar_file = tarfile.open(data_dir, 'r')\n",
        "        self.filenames = [name for name in self.tar_file.getnames() if name.endswith('.png')]\n",
        "\n",
        "        self.transform = transforms.Compose([\n",
        "            transforms.Grayscale(),        # Convert to grayscale\n",
        "            transforms.ToTensor(),         # Convert PIL Image to PyTorch Tensor\n",
        "            transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std\n",
        "        ])\n",
        "\n",
        "        # Extract labels from filenames\n",
        "        self.labels = [int(os.path.basename(name).split('.')[0]) for name in self.filenames]\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.filenames)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        filename = self.filenames[idx]\n",
        "        label = self.labels[idx]\n",
        "\n",
        "        # Extract image from tar file\n",
        "        file_content = self.tar_file.extractfile(filename).read()\n",
        "        image = Image.open(io.BytesIO(file_content))\n",
        "        image = self.transform(image)\n",
        "\n",
        "        return image, label\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    # Test the implementation\n",
        "    train_dataset = MNIST(data_dir='/content/drive/MyDrive/SeoulTech/인공신경망과 딥러닝/train.tar')\n",
        "    train_image, train_label = train_dataset[0]\n",
        "    print(f\"Train image shape: {train_image.shape}\")\n",
        "    print(f\"Train label: {train_label}\")\n"
      ],
      "metadata": {
        "id": "VgzL31MGhfbr",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b32c5baf-b5f2-437f-b0cb-64ebd955dbbc"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Train image shape: torch.Size([1, 28, 28])\n",
            "Train label: 408583\n"
          ]
        }
      ]
    }
  ]
}