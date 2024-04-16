{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1Pp_dLu76dmM-WDWmjLe-hGdr6HKZxnrU",
      "authorship_tag": "ABX9TyOj56EDEeTfehGeQ9EUbt8y",
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
        "!pip install torch torchvision"
      ],
      "metadata": {
        "id": "ZMKUIgi8WVQc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "2W7kMx3UFKuT"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "from PIL import Image\n",
        "import cv2\n",
        "from torch.utils.data import Dataset\n",
        "from torchvision import transforms\n",
        "\n",
        "class MNIST(Dataset):\n",
        "    \"\"\" MNIST dataset\n",
        "\n",
        "    Args:\n",
        "        data_dir: directory path containing images\n",
        "\n",
        "    Note:\n",
        "        1) Each image should be preprocessed as follows:\n",
        "            - First, all values should be in a range of [0,1]\n",
        "            - Subtract mean of 0.1307, and divide by std 0.3081\n",
        "            - These preprocessing can be implemented using torchvision.transforms\n",
        "        2) Labels can be obtained from filenames: {number}_{label}.png\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(self, data_dir, transform=None):\n",
        "        self.data_dir = data_dir\n",
        "        self.transform = transform\n",
        "        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.png')]\n",
        "        self.image_files = os.listdir(data_dir)\n",
        "        self.transform = transforms.Compose([\n",
        "            transforms.ToTensor(),  # PIL Image 또는 numpy array를 토치 텐서로 변환\n",
        "            transforms.Normalize((0.1307,), (0.3081,))  # 평균과 표준편차로 정규화\n",
        "        ])\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.image_files)\n",
        "        return len(self.filenames)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        image_file = self.image_files[idx]\n",
        "        image_path = os.path.join(self.data_dir, image_file)\n",
        "        img_name = os.path.join(self.data_dir, self.filenames[idx])\n",
        "        image = Image.open(img_name)\n",
        "        label = int(self.filenames[idx].split('_')[1].split('.')[0])\n",
        "\n",
        "        if self.transform:\n",
        "            image = self.transform(image)\n",
        "        # 이미지 읽어들이기 (예시로는 OpenCV 사용)\n",
        "        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)\n",
        "        # 이미지를 MNIST 데이터셋과 유사한 크기(28x28)로 리사이즈\n",
        "        image = cv2.resize(image, (28, 28))\n",
        "        # 이미지를 0~1 사이 값으로 정규화\n",
        "        image = image / 255.0\n",
        "        # 텐서로 변환 및 정규화\n",
        "        image = self.transform(image)\n",
        "        # 라벨 추출 (파일 이름에서 추출)\n",
        "        label = int(image_file.split(\"_\")[1].split(\".\")[0])  # 파일 이름 형식에 따라 변경 필요\n",
        "        return image, label\n"
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
        "outputId": "b7ca2428-6b21-43f0-af71-6ac2134f11b7"
      },
      "execution_count": 5,
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
        "outputId": "b3875281-2247-4d2b-cc73-2287eb46d49a"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "압축 해제된 데이터 디렉토리 내 파일 목록: ['mnist-classification']\n"
          ]
        }
      ]
    }
  ]
}