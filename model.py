{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMGUY410pkQTijLp7gZvBG4",
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
      "cell_type": "markdown",
      "source": [
        "Tensorflow Keras로 CNN모델 코드 구현"
      ],
      "metadata": {
        "id": "Yl2djO2SjjUP"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "데이터셋 **로드**\n",
        "\n",
        "먼저, keras에서 자체적으로 제공하는 MNIST 데이터셋을 가져오고, shape를 확인해보겠습니다.\n",
        "참고로, MNIST 데이터셋의 각 이미지는 28 * 28 픽셀 크기이며, RGB 차원은 따로 없는 흑백 사진으로 구성되어 있습니다.\n",
        "6만 장의 학습용 데이터와 1만 장의 테스트용 데이터로 구성되어 있으며, 라벨의 종류는 0~9의 숫자로 이루어진 10 종류입니다.\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "sADX80s-jgmH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf\n",
        "\n",
        "mnist = tf.keras.datasets.mnist\n",
        "(X_train,y_train), (X_test,y_test) = mnist.load_data()\n",
        "\n",
        "# 0 ~ 255 사이 정수 -> 0 ~ 1 사이 실수로 변환\n",
        "X_train,X_test = X_train / 255.0, X_test / 255.0\n",
        "\n",
        "print(\"X_train shape :\", X_train.shape)  #X_train shape : (60000, 28, 28)]\n",
        "print(\"y_train shape :\", y_train.shape) #y_train shape : (60000,)\n",
        "print(\"X_test shape :\", X_test.shape) #X_test shape : (10000, 28, 28)\n",
        "print(\"y_test shape :\", y_test.shape) #y_test shape : (10000,)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LSQlWt_fdr7r",
        "outputId": "0880ec33-3936-4547-8189-ee2af754194b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "X_train shape : (60000, 28, 28)\n",
            "y_train shape : (60000,)\n",
            "X_test shape : (10000, 28, 28)\n",
            "y_test shape : (10000,)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "CNN 모델 layer **쌓기**"
      ],
      "metadata": {
        "id": "8x_ak7Qyjchp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras import models, layers\n",
        "\n",
        "model = models.Sequential()\n",
        "\n",
        "# 필터 5개, 커널 사이즈 = 3 * 3인 convolution layer - maxpooling 조합\n",
        "model.add(layers.Conv2D(5, 3, strides = 1, padding = 'same', activation = 'relu',\n",
        "input_shape = (28, 28, 1))) #RGB가 있는 경우는 마지막 숫자가 3이어야 함.\n",
        "model.add(layers.MaxPooling2D(pool_size=(2,2)))\n",
        "\n",
        "# 필터 10개, 커널 사이즈 = 3 * 3인 convolution layer - maxpooling 조합\n",
        "model.add(layers.Conv2D(10, 3, strides = 1, padding = 'same', activation = 'relu'))\n",
        "model.add(layers.MaxPooling2D(pool_size=(2, 2)))\n",
        "\n",
        "# 1차원으로 변환 후 fc layer 통과(64차원 변환 -> dropout -> 10개 클래스 확률 변환)\n",
        "model.add(layers.Flatten())\n",
        "model.add(layers.Dense(64, activation = 'relu'))\n",
        "model.add(layers.Dropout(0,2))\n",
        "model.add(layers.Dense(10, activation = 'softmax'))  # 클래스 10개, softmax 적용하여 각 클래스의 확률로 변환\n",
        "\n",
        "#마지막으로 Dense layer부분에서는 유닛 개수를 클래스 개수로 정해야하고, 활성함수를 softmax로 지정해햐 한다는 점.\n",
        "# (단, 이진 분류에서는 유닛 개수 = 1, 활성 함수 = sigmoid로 지정해야함.)"
      ],
      "metadata": {
        "id": "5ot7wdfngnkF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**CNN 모델 학습**"
      ],
      "metadata": {
        "id": "8Z4cElVfjUNz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])\n",
        "\n",
        "history = model.fit(X_train, y_train, epochs = 5, batch_size = 16, verbose = 1, validation_data = [X_test, y_test])\n",
        "\n",
        "#verbose = 1로 지정하여 학습 결과 창이 위와 같이 출력되도록 하였으며\n",
        "#validation data도 지정하여 테스트 데이터에서의 정확도 성능도 같이 평가한 결과,\n",
        "#98.5% 이상의 테스트셋 정확도를 기록한 점을 알 수 있었습니다."
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NaBY6YL1inxr",
        "outputId": "02fd4e5f-149c-44d1-ae59-95a8a7bf2c1a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/5\n",
            "3750/3750 [==============================] - 49s 12ms/step - loss: 0.1874 - accuracy: 0.9443 - val_loss: 0.0725 - val_accuracy: 0.9762\n",
            "Epoch 2/5\n",
            "3750/3750 [==============================] - 51s 14ms/step - loss: 0.0711 - accuracy: 0.9776 - val_loss: 0.0485 - val_accuracy: 0.9833\n",
            "Epoch 3/5\n",
            "3750/3750 [==============================] - 51s 14ms/step - loss: 0.0517 - accuracy: 0.9838 - val_loss: 0.0412 - val_accuracy: 0.9863\n",
            "Epoch 4/5\n",
            "3750/3750 [==============================] - 47s 13ms/step - loss: 0.0408 - accuracy: 0.9876 - val_loss: 0.0393 - val_accuracy: 0.9877\n",
            "Epoch 5/5\n",
            "3750/3750 [==============================] - 45s 12ms/step - loss: 0.0338 - accuracy: 0.9894 - val_loss: 0.0471 - val_accuracy: 0.9852\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "0jTbmgxTia3h"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "W-fgTxt8iXaD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "9HqSgyTEgiGq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# example of loading the mnist dataset\n",
        "from tensorflow.keras.datasets import mnist\n",
        "from matplotlib import pyplot as plt\n",
        "# load dataset\n",
        "(trainX, trainy), (testX, testy) = mnist.load_data()\n",
        "# summarize loaded dataset\n",
        "print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))\n",
        "print('Test: X=%s, y=%s' % (testX.shape, testy.shape))\n",
        "# plot first few images\n",
        "for i in range(9):\n",
        "\t# define subplot\n",
        "\tplt.subplot(330 + 1 + i)\n",
        "\t# plot raw pixel data\n",
        "\tplt.imshow(trainX[i], cmap=plt.get_cmap('gray'))\n",
        "# show the figure\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 470
        },
        "id": "LNwEKlAIDmZK",
        "outputId": "3d05c1b1-8b4f-45c9-c814-e536c4744cf7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Train: X=(60000, 28, 28), y=(60000,)\n",
            "Test: X=(10000, 28, 28), y=(10000,)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 9 Axes>"
            ],
            "image/png": "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MmjQIN/6O++8o2zTj5/px+j1Y64LFiwIadtCzePxSHx8vNHNMEx31+6TTz6p5KeeekrJLS0tSv7ggw+UfMcddyj5+PHj7b6WvsPQnyf/u9/9zu/99ceX6NtqNGrXXn93g/H6668rWT/fid7YsWOVrL8ORLh1pnYNP+oeAACEDx09AAA2RkcPAICNcT36MLvhhht865dffrnf+7Y9PUZE5KWXXgpLm2BNCQkJSp41a5aS9Yfb6MfkAz0Cu+11ud9++21lW0ZGht/H/uEPf1DywoULA3ptIBht52no06dPQI/93ve+53f7xx9/rOT2phk2E/boAQCwMTp6AABsjK/uQ0z/9ejPf/7zdu+rPw1DfxqHx+MJWbtgfbGxsUrWT6msp59mNikpScn33nuvkidOnKjkYcOG+db79u2rbNMPE+jzW2+9peSmpia/bQX86d27t5IvvfRSJetP15wwYUK7zxUdre7f6k9r1jt48KCS9b83p0+f9vt4M2CPHgAAG6OjBwDAxujoAQCwMcbog9R2iluRM6e59efvf/+7kt1udyiaBJvST2mrv7Tr+eefr+R9+/YpOdDZrtuOTeovWdv2cqgiIkeOHFHyu+++G9BrIbLFxMQo+fvf/76S9X9X9fWnn765be3qT39re8qzyJnj/3rnnKN2k5MnT1ay/jRo/e+pGbBHDwCAjdHRAwBgY3T0AADYGGP0QfrpT3+q5I7OyWzL3zn2gF5DQ4OS9XM2bNy4UcmJiYlK/vLLL5W8YcMGJZeWlir5X//6l2999erVyjb9GKl+O+CPfk4I/bj5H//4R7+Pf/rpp5VcXl6u5I8++si3rv890N+37XwRZ6M/9qW4uFjJBw4cUPL69euV3Nzc7Pf5uwN79AAA2BgdPQAANkZHDwCAjTFGH6ArrrhCyePGjev0Y/VjorW1taFoEiLUrl27lKwfSwzWmDFjfOtXX321sk1/LIp+TgigLf158vox9kceecTv499//30lL168WMn641fa/i78+c9/VrbpL0OrP+9df0ll/Rj+zTffrGT9JZz/8z//U8kvvPCCkr/55htpz+7du9vdFgz26AEAsDE6egAAbIyOHgAAG2OMPkAffvihks8991y/99+5c6dvfdq0aeFoEhAWvXr18q3rx+T18+ZzHj3a6tGjh5KfeeYZJf/kJz9RclNTk5J/9rOfKVlf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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# load dataset\n",
        "(trainX, trainY), (testX, testY) = mnist.load_data()\n",
        "# reshape dataset to have a single channel\n",
        "trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))\n",
        "testX = testX.reshape((testX.shape[0], 28, 28, 1))"
      ],
      "metadata": {
        "id": "_e78IC3aEciR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# one hot encode target values\n",
        "trainY = to_categorical(trainY)\n",
        "testY = to_categorical(testY)"
      ],
      "metadata": {
        "id": "CuMrAMT6E5xm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from keras.utils import to_categorical"
      ],
      "metadata": {
        "id": "XA0QusBYFHk1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# load train and test dataset\n",
        "def load_dataset():\n",
        "\t# load dataset\n",
        "\t(trainX, trainY), (testX, testY) = mnist.load_data()\n",
        "\t# reshape dataset to have a single channel\n",
        "\ttrainX = trainX.reshape((trainX.shape[0], 28, 28, 1))\n",
        "\ttestX = testX.reshape((testX.shape[0], 28, 28, 1))\n",
        "\t# one hot encode target values\n",
        "\ttrainY = to_categorical(trainY)\n",
        "\ttestY = to_categorical(testY)\n",
        "\treturn trainX, trainY, testX, testY"
      ],
      "metadata": {
        "id": "uwEUIuuXFHZe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# convert from integers to floats\n",
        "train_norm = train.astype('float32')\n",
        "test_norm = test.astype('float32')\n",
        "# normalize to range 0-1\n",
        "train_norm = train_norm / 255.0\n",
        "test_norm = test_norm / 255.0"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 221
        },
        "id": "kPdeLkK0Fb0a",
        "outputId": "5cbe7fd0-50d3-476b-aef6-f0853c56cda4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'train' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-32-2eff0501d7ea>\u001b[0m in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# convert from integers to floats\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mtrain_norm\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrain\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mastype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'float32'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0mtest_norm\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtest\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mastype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'float32'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;31m# normalize to range 0-1\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mtrain_norm\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrain_norm\u001b[0m \u001b[0;34m/\u001b[0m \u001b[0;36m255.0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'train' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "u3Qm3_lOGR5g"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "ufxPyjPXGSeR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "LczqOj2rGS8w"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "Z2Z021W7GTJy"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "V34v3dY2GTTd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "CMelDejLGPVe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# scale pixels\n",
        "def prep_pixels(train, test):\n",
        "\t# convert from integers to floats\n",
        "\ttrain_norm = train.astype('float32')\n",
        "\ttest_norm = test.astype('float32')\n",
        "\t# normalize to range 0-1\n",
        "\ttrain_norm = train_norm / 255.0\n",
        "\ttest_norm = test_norm / 255.0\n",
        "\t# return normalized images\n",
        "\treturn train_norm, test_norm"
      ],
      "metadata": {
        "id": "-tRDCRIyGGJw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# define cnn model\n",
        "def define_model():\n",
        "\tmodel = Sequential()\n",
        "\tmodel.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))\n",
        "\tmodel.add(MaxPooling2D((2, 2)))\n",
        "\tmodel.add(Flatten())\n",
        "\tmodel.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))\n",
        "\tmodel.add(Dense(10, activation='softmax'))\n",
        "\t# compile model\n",
        "\topt = SGD(learning_rate=0.01, momentum=0.9)\n",
        "\tmodel.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])\n",
        "\treturn model"
      ],
      "metadata": {
        "id": "ccXrDGViGF-q"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# evaluate a model using k-fold cross-validation\n",
        "def evaluate_model(dataX, dataY, n_folds=5):\n",
        "\tscores, histories = list(), list()\n",
        "\t# prepare cross validation\n",
        "\tkfold = KFold(n_folds, shuffle=True, random_state=1)\n",
        "\t# enumerate splits\n",
        "\tfor train_ix, test_ix in kfold.split(dataX):\n",
        "\t\t# define model\n",
        "\t\tmodel = define_model()\n",
        "\t\t# select rows for train and test\n",
        "\t\ttrainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]\n",
        "\t\t# fit model\n",
        "\t\thistory = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)\n",
        "\t\t# evaluate model\n",
        "\t\t_, acc = model.evaluate(testX, testY, verbose=0)\n",
        "\t\tprint('> %.3f' % (acc * 100.0))\n",
        "\t\t# stores scores\n",
        "\t\tscores.append(acc)\n",
        "\t\thistories.append(history)\n",
        "\treturn scores, histories"
      ],
      "metadata": {
        "id": "ThJrPaYjGi2p"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# plot diagnostic learning curves\n",
        "def summarize_diagnostics(histories):\n",
        "\tfor i in range(len(histories)):\n",
        "\t\t# plot loss\n",
        "\t\tplt.subplot(2, 1, 1)\n",
        "\t\tplt.title('Cross Entropy Loss')\n",
        "\t\tplt.plot(histories[i].history['loss'], color='blue', label='train')\n",
        "\t\tplt.plot(histories[i].history['val_loss'], color='orange', label='test')\n",
        "\t\t# plot accuracy\n",
        "\t\tplt.subplot(2, 1, 2)\n",
        "\t\tplt.title('Classification Accuracy')\n",
        "\t\tplt.plot(histories[i].history['accuracy'], color='blue', label='train')\n",
        "\t\tplt.plot(histories[i].history['val_accuracy'], color='orange', label='test')\n",
        "\tplt.show()"
      ],
      "metadata": {
        "id": "d_nYsN2fGlUH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# summarize model performance\n",
        "def summarize_performance(scores):\n",
        "\t# print summary\n",
        "\tprint('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))\n",
        "\t# box and whisker plots of results\n",
        "\tplt.boxplot(scores)\n",
        "\tplt.show()"
      ],
      "metadata": {
        "id": "JHJ2DnhVGoNR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# run the test harness for evaluating a model\n",
        "def run_test_harness():\n",
        "\t# load dataset\n",
        "\ttrainX, trainY, testX, testY = load_dataset()\n",
        "\t# prepare pixel data\n",
        "\ttrainX, testX = prep_pixels(trainX, testX)\n",
        "\t# evaluate model\n",
        "\tscores, histories = evaluate_model(trainX, trainY)\n",
        "\t# learning curves\n",
        "\tsummarize_diagnostics(histories)\n",
        "\t# summarize estimated performance\n",
        "\tsummarize_performance(scores)"
      ],
      "metadata": {
        "id": "vmUtOSmfGnfe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# baseline cnn model for mnist\n",
        "from numpy import mean\n",
        "from numpy import std\n",
        "from matplotlib import pyplot as plt\n",
        "from sklearn.model_selection import KFold\n",
        "from tensorflow.keras.datasets import mnist\n",
        "from tensorflow.keras.utils import to_categorical\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Conv2D\n",
        "from tensorflow.keras.layers import MaxPooling2D\n",
        "from tensorflow.keras.layers import Dense\n",
        "from tensorflow.keras.layers import Flatten\n",
        "from tensorflow.keras.optimizers import SGD\n",
        "\n",
        "# load train and test dataset\n",
        "def load_dataset():\n",
        "\t# load dataset\n",
        "\t(trainX, trainY), (testX, testY) = mnist.load_data()\n",
        "\t# reshape dataset to have a single channel\n",
        "\ttrainX = trainX.reshape((trainX.shape[0], 28, 28, 1))\n",
        "\ttestX = testX.reshape((testX.shape[0], 28, 28, 1))\n",
        "\t# one hot encode target values\n",
        "\ttrainY = to_categorical(trainY)\n",
        "\ttestY = to_categorical(testY)\n",
        "\treturn trainX, trainY, testX, testY\n",
        "\n",
        "# scale pixels\n",
        "def prep_pixels(train, test):\n",
        "\t# convert from integers to floats\n",
        "\ttrain_norm = train.astype('float32')\n",
        "\ttest_norm = test.astype('float32')\n",
        "\t# normalize to range 0-1\n",
        "\ttrain_norm = train_norm / 255.0\n",
        "\ttest_norm = test_norm / 255.0\n",
        "\t# return normalized images\n",
        "\treturn train_norm, test_norm\n",
        "\n",
        "# define cnn model\n",
        "def define_model():\n",
        "\tmodel = Sequential()\n",
        "\tmodel.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))\n",
        "\tmodel.add(MaxPooling2D((2, 2)))\n",
        "\tmodel.add(Flatten())\n",
        "\tmodel.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))\n",
        "\tmodel.add(Dense(10, activation='softmax'))\n",
        "\t# compile model\n",
        "\topt = SGD(learning_rate=0.01, momentum=0.9)\n",
        "\tmodel.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])\n",
        "\treturn model\n",
        "\n",
        "# evaluate a model using k-fold cross-validation\n",
        "def evaluate_model(dataX, dataY, n_folds=5):\n",
        "\tscores, histories = list(), list()\n",
        "\t# prepare cross validation\n",
        "\tkfold = KFold(n_folds, shuffle=True, random_state=1)\n",
        "\t# enumerate splits\n",
        "\tfor train_ix, test_ix in kfold.split(dataX):\n",
        "\t\t# define model\n",
        "\t\tmodel = define_model()\n",
        "\t\t# select rows for train and test\n",
        "\t\ttrainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]\n",
        "\t\t# fit model\n",
        "\t\thistory = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)\n",
        "\t\t# evaluate model\n",
        "\t\t_, acc = model.evaluate(testX, testY, verbose=0)\n",
        "\t\tprint('> %.3f' % (acc * 100.0))\n",
        "\t\t# stores scores\n",
        "\t\tscores.append(acc)\n",
        "\t\thistories.append(history)\n",
        "\treturn scores, histories\n",
        "\n",
        "# plot diagnostic learning curves\n",
        "def summarize_diagnostics(histories):\n",
        "\tfor i in range(len(histories)):\n",
        "\t\t# plot loss\n",
        "\t\tplt.subplot(2, 1, 1)\n",
        "\t\tplt.title('Cross Entropy Loss')\n",
        "\t\tplt.plot(histories[i].history['loss'], color='blue', label='train')\n",
        "\t\tplt.plot(histories[i].history['val_loss'], color='orange', label='test')\n",
        "\t\t# plot accuracy\n",
        "\t\tplt.subplot(2, 1, 2)\n",
        "\t\tplt.title('Classification Accuracy')\n",
        "\t\tplt.plot(histories[i].history['accuracy'], color='blue', label='train')\n",
        "\t\tplt.plot(histories[i].history['val_accuracy'], color='orange', label='test')\n",
        "\tplt.show()\n",
        "\n",
        "# summarize model performance\n",
        "def summarize_performance(scores):\n",
        "\t# print summary\n",
        "\tprint('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))\n",
        "\t# box and whisker plots of results\n",
        "\tplt.boxplot(scores)\n",
        "\tplt.show()\n",
        "\n",
        "# run the test harness for evaluating a model\n",
        "def run_test_harness():\n",
        "\t# load dataset\n",
        "\ttrainX, trainY, testX, testY = load_dataset()\n",
        "\t# prepare pixel data\n",
        "\ttrainX, testX = prep_pixels(trainX, testX)\n",
        "\t# evaluate model\n",
        "\tscores, histories = evaluate_model(trainX, trainY)\n",
        "\t# learning curves\n",
        "\tsummarize_diagnostics(histories)\n",
        "\t# summarize estimated performance\n",
        "\tsummarize_performance(scores)\n",
        "\n",
        "# entry point, run the test harness\n",
        "run_test_harness()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 975
        },
        "id": "n1b7pTYQGsWQ",
        "outputId": "fd2b0960-6695-4969-db6f-2ba14bf3e911"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "> 98.550\n",
            "> 98.675\n",
            "> 98.500\n",
            "> 98.550\n",
            "> 98.767\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiwAAAGzCAYAAAAMr0ziAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACTAklEQVR4nOzdd3xT9f7H8VeSNkl3gS4KhbL3HhVRcCAo6FUvKnBREK8bUKwLHODCCl79oeAV5aog48pwXReKOFFkioAs2bMtBbp3cn5/fDO7U9omtJ/n43EeOefk5OSbdOSd7zo6TdM0hBBCCCF8mN7bBRBCCCGEqIwEFiGEEEL4PAksQgghhPB5EliEEEII4fMksAghhBDC50lgEUIIIYTPk8AihBBCCJ8ngUUIIYQQPk8CixBCCCF8ngQWIYQQQvg8CSxCeMmBAwe45557aN26NWazmdDQUAYOHMhrr71GXl6et4tXJc888ww6na7cJTk52eNzLlu2jDlz5tR8YetQfHw81157rbeLIUS94uftAgjREH3xxRfcfPPNmEwmxo0bR9euXSksLGTdunU8+uij/Pnnn7z99tveLmaVvfnmmwQHB5faHx4e7vG5li1bxs6dO5kyZcr5F0wIUW9IYBGijh06dIjRo0fTsmVLvvvuO5o2beq4b+LEiezfv58vvvii3MdbrVYKCwsxm811Udwquemmm4iIiKjz583Pz8doNKLXS2WxEPWd/JULUcdmz55NdnY277zzjltYsWvbti0PPvigY1un0zFp0iSWLl1Kly5dMJlMrF69GoDff/+da665htDQUIKDg7nyyiv57bff3M5XVFTEs88+S7t27TCbzTRp0oRLLrmENWvWOI5JTk5mwoQJNG/eHJPJRNOmTbn++us5fPhwjbzmH374AZ1Ox4oVK5g5cybNmzfHbDZz5ZVXsn//fsdxl112GV988QVHjhxxNCvFx8e7neODDz7gqaeeolmzZgQGBpKZmQnAypUr6dOnDwEBAURERHDrrbdy4sQJt3LcfvvtBAcHc/DgQYYNG0ZQUBCxsbE899xz2C9cr2ka8fHxXH/99aVeR35+PmFhYdxzzz3n/Z4UFxfz/PPP06ZNG0wmE/Hx8TzxxBMUFBS4Hbd582aGDRtGREQEAQEBtGrVijvuuMPtmA8++IA+ffoQEhJCaGgo3bp147XXXjvvMgrhS6SGRYg69tlnn9G6dWsuvvjiKj/mu+++Y8WKFUyaNImIiAji4+P5888/ufTSSwkNDeWxxx7D39+ft956i8suu4wff/yRhIQEQPUzSUpK4s4776R///5kZmayefNmtm7dylVXXQXAyJEj+fPPP5k8eTLx8fGkpqayZs0ajh496ggMFTl79mypfX5+fqWahF566SX0ej2PPPIIGRkZzJ49m7Fjx7JhwwYAnnzySTIyMjh+/Dj/93//B1Cqqen555/HaDTyyCOPUFBQgNFoZOHChUyYMIF+/fqRlJRESkoKr732Gr/88gu///67WzksFgtXX301F110EbNnz2b16tXMmDGD4uJinnvuOXQ6HbfeeiuzZ8/m7NmzNG7c2PHYzz77jMzMTG699dZK35PK3HnnnSxatIibbrqJhx9+mA0bNpCUlMTu3bv5+OOPAUhNTWXo0KFERkYydepUwsPDOXz4MB999JHjPGvWrGHMmDFceeWVzJo1C4Ddu3fzyy+/uAVfIS54mhCizmRkZGiAdv3111f5MYCm1+u1P//8023/DTfcoBmNRu3AgQOOfSdPntRCQkK0QYMGOfb16NFDGzFiRLnnP3funAZoL7/8ctVfiM2MGTM0oMylQ4cOjuO+//57DdA6deqkFRQUOPa/9tprGqDt2LHDsW/EiBFay5YtSz2X/RytW7fWcnNzHfsLCwu1qKgorWvXrlpeXp5j/+eff64B2vTp0x37xo8frwHa5MmTHfusVqs2YsQIzWg0aqdPn9Y0TdP27t2rAdqbb77pVoa//e1vWnx8vGa1Wit8X1q2bFnhe75t2zYN0O688063/Y888ogGaN99952maZr28ccfa4C2adOmcs/14IMPaqGhoVpxcXGFZRLiQidNQkLUIXvzRUhIiEePGzx4MJ07d3ZsWywWvvnmG2644QZat27t2N+0aVP+8Y9/sG7dOsdzhYeH8+eff/LXX3+Vee6AgACMRiM//PAD586d8/QlAfDhhx+yZs0at+W9994rddyECRMwGo2O7UsvvRSAgwcPVvm5xo8fT0BAgGN78+bNpKamcv/997v16xkxYgQdO3Yssz/QpEmTHOv2JrfCwkK+/fZbANq3b09CQgJLly51HHf27Fm++uorxo4di06nq3J5y/Lll18CkJiY6Lb/4YcfBnCU2V4z9Pnnn1NUVFTmucLDw8nJyXFr4hOiPpLAIkQdCg0NBSArK8ujx7Vq1cpt+/Tp0+Tm5tKhQ4dSx3bq1Amr1cqxY8cAeO6550hPT6d9+/Z069aNRx99lO3btzuON5lMzJo1i6+++oro6GgGDRrE7NmzPRqSPGjQIIYMGeK2DBgwoNRxLVq0cNtu1KgRgEdBqeR7ceTIEYAy34uOHTs67rfT6/VuIQ9UQAHc+uyMGzeOX375xfH4lStXUlRUxG233VblspbnyJEj6PV62rZt67Y/JiaG8PBwx3MOHjyYkSNH8uyzzxIREcH111/Pe++959bP5f7776d9+/Zcc801NG/enDvuuMPRx0mI+kQCixB1KDQ0lNjYWHbu3OnR41xrFDw1aNAgDhw4wLvvvkvXrl35z3/+Q+/evfnPf/7jOGbKlCns27ePpKQkzGYzTz/9NJ06deL333+v9vOWxWAwlLlfs3V4rYrzeS88MXr0aPz9/R21LEuWLKFv375lBqPqqqymRqfTsWrVKtavX8+kSZM4ceIEd9xxB3369CE7OxuAqKgotm3bxv/+9z/+9re/8f3333PNNdcwfvz4GiunEL5AAosQdezaa6/lwIEDrF+/vtrniIyMJDAwkL1795a6b8+ePej1euLi4hz7GjduzIQJE/jvf//LsWPH6N69O88884zb49q0acPDDz/MN998w86dOyksLOSVV16pdhmry9PmlpYtWwKU+V7s3bvXcb+d1Wot1QS1b98+ALcOxo0bN2bEiBEsXbqUI0eO8Msvv9RI7Yq9zFartVQzXUpKCunp6aXKfNFFFzFz5kw2b97M0qVL+fPPP/nggw8c9xuNRq677jr+/e9/OyYkfP/9991GYAlxoZPAIkQde+yxxwgKCuLOO+8kJSWl1P0HDhyodEiqwWBg6NChfPrpp27NGCkpKSxbtoxLLrnE0fx05swZt8cGBwfTtm1bR7NCbm4u+fn5bse0adOGkJCQUkNs60JQUBAZGRlVPr5v375ERUUxf/58t/J+9dVX7N69mxEjRpR6zLx58xzrmqYxb948/P39ufLKK92Ou+2229i1axePPvooBoOB0aNHV+MVlTZ8+HCAUjP6vvrqqwCOMp87d65U7VPPnj0BHK+15M9Xr9fTvXt3t2OEqA9kWLMQdaxNmzYsW7aMUaNG0alTJ7eZbn/99VdWrlzJ7bffXul5XnjhBdasWcMll1zC/fffj5+fH2+99RYFBQXMnj3bcVznzp257LLL6NOnD40bN2bz5s2sWrXK0fF03759XHnlldxyyy107twZPz8/Pv74Y1JSUqr8Ab1q1aoyZ7q96qqriI6OrtobY9OnTx+WL19OYmIi/fr1Izg4mOuuu67c4/39/Zk1axYTJkxg8ODBjBkzxjGsOT4+noceesjteLPZzOrVqxk/fjwJCQl89dVXfPHFFzzxxBNERka6HTtixAiaNGnCypUrueaaa4iKiqry69i/fz8vvPBCqf29evVixIgRjB8/nrfffpv09HQGDx7Mxo0bWbRoETfccAOXX345AIsWLeLf//43N954I23atCErK4sFCxYQGhrqCD133nknZ8+e5YorrqB58+YcOXKEuXPn0rNnTzp16lTl8grh87w8SkmIBmvfvn3aXXfdpcXHx2tGo1ELCQnRBg4cqM2dO1fLz893HAdoEydOLPMcW7du1YYNG6YFBwdrgYGB2uWXX679+uuvbse88MILWv/+/bXw8HAtICBA69ixozZz5kytsLBQ0zRNS0tL0yZOnKh17NhRCwoK0sLCwrSEhARtxYoVlb6GioY1A9r333+vaZpzSPLKlSvdHn/o0CEN0N577z3HvuzsbO0f//iHFh4ergGOIc7lncNu+fLlWq9evTSTyaQ1btxYGzt2rHb8+HG3Y8aPH68FBQVpBw4c0IYOHaoFBgZq0dHR2owZMzSLxVLmee+//34N0JYtW1bp+2HXsmXLct+Tf/7zn5qmaVpRUZH27LPPaq1atdL8/f21uLg4bdq0aW4/+61bt2pjxozRWrRooZlMJi0qKkq79tprtc2bNzuOWbVqlTZ06FAtKipKMxqNWosWLbR77rlHO3XqVJXLK8SFQKdpHvR2E0KIC9jtt9/OqlWrHB1Wq+Khhx7inXfeITk5mcDAwFosnRCiItKHRQghypGfn8+SJUsYOXKkhBUhvEz6sAghRAmpqal8++23rFq1ijNnzsgU90L4AAksQghRwq5duxg7dixRUVG8/vrrjpE5QgjvkT4sQgghhPB50odFCCGEED5PAosQQgghfF696cNitVo5efIkISEh530lVSGEEELUDU3TyMrKIjY2Fr2+/HqUehNYTp486XbtFCGEEEJcOI4dO0bz5s3Lvb/eBJaQkBBAvWD7NVSEEEII4dsyMzOJ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ffv22hNPPKEdPXpUi4qK0r766ivtvvvu09q3b6/p9XrNbDZrN910k3bo0CHHOd57770yn/f777/XNE3TBg8erA0ePNjteVNSUrQ77rhDi4qK0kwmk9a9e3dt4cKFbsfYy/zyyy9rb731lta6dWvNaDRqffv21TZu3Filn4+madrixYs1vV6vnTp1Sps1a5YWGhqq5eXllTouLy9PmzFjhtauXTvNZDJpMTEx2o033qjt37/fcYzFYtHmzJmjde3aVTOZTFpERIQ2bNgwbdOmTW5lLvk+a5qmAdqMGTMc2zNmzNAA7c8//9TGjBmjhYeHaz179tQ0TdP++OMPbfz48VqrVq00k8mkAVpkZKSWlpbmePzkyZO1iy66SDt+/Lh2xx13aE2bNtWMRqMWHx+v3XvvvVpBQYF24MABDdBeffXVUuX55ZdfNEBbtmxZld9L4fTf//5Xa968ufbf//5X2759u/b+++9rjRs3LvV7LKpOaljKERERgcFgICUlxW1/SkoKMTExXiqVmDRpEp9//jk//fQTzZs3r/Pn/+yzz2jdujUXX3xxtR5/8OBBPvnkE26++WZatWpFSkoKb731FoMHD2bXrl3ExsYCqinngQce4KabbuLBBx8kPz+f7du3s2HDBv7xj38AcO+997Jq1SomTZpE586dOXPmDOvWrWP37t307t271HN36tSJxYsX89BDD9G8eXMefvhhACIjIzl9+nSp47dv386ll16Kv78/d999N/Hx8Rw4cIDPPvuMfv36kZqayogRI7Bareh0OjRNIz8/n1WrVrFp0yZ27dpFYGAggwYN4oEHHuD111/niSeeoFOnTo7ylCUvL4/LLruM/fv3M2nSJFq1asXKlSu5/fbbSU9PL9XktmzZMrKysrjnnnvQ6XTMnj2bv//97xw8eLDMGqaSli5dyuWXX05MTAyjR49m6tSpfPbZZ9x8882OYywWC9deey1r165l9OjRPPjgg2RlZbFmzRp27tzp6Cvyz3/+k4ULF3LNNddw5513UlxczM8//8xvv/1G3759Ky1LWW6++WbatWvHiy++6BhhsmbNGg4ePMiECROIiYnh4Ycf5uzZswwfPpzffvsNnU5Hp06dWLFiBf379yc9PZ27776bjh07cuLECVatWkVubi6tW7dm4MCBLF26lIceeqjU+xISEsL1119frXI3dI8+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232u5S/r9cydO7dUjYF97pCSH9RlGT58OMnJyW7NfcXFxcydO5fg4GAGDx7s6csp09KlS7n00ksZNWoUN910k9vy6KOPAvDf//4XgJEjR5KWlsa8efNKncf++keOHImmaTz77LPlHhMaGkpERAQ//fST2/3//ve/q1xue7gq+b7PmTPHbVuv13PDDTfw2WeflXlNIdfH+/n5MWbMGFasWMHChQvp1q1btS+1IERtkRoWITzQpk0bli1bxqhRo+jUqRPjxo2ja9euFBYW8uuvvzqG35bn2muv5bnnnmPChAlcfPHF7Nixg6VLlzpqNOyGDh1KTEwMAwcOJDo6mt27dzNv3jxGjBhBSEgI6enpNG/enJtuuokePXoQHBzMt99+y6ZNm9xqO87H66+/ziWXXELv3r25++67adWqFYcPH+aLL75g27ZtjtezePFiwsLC6Ny5M+vXr+fbb7+lSZMmbufq2bMnBoOBWbNmkZGRgclk4oorriAqKqrU895999289dZb3H777WzZsoX4+HhWrVrFL7/8wpw5c86r07Pdhg0bHMOmy9KsWTN69+7N0qVLefzxxxk3bhzvv/8+iYmJbNy4kUsvvZScnBy+/fZb7r//fq6//nouv/xybrvtNl5//XX++usvrr76aqxWKz///DOXX36547nuvPNOXnrpJe6880769u3LTz/95JgNtSpCQ0MZNGgQs2fPpqioiGbNmvHNN99w6NChUse++OKLfPPNNwwePJi7776bTp06cerUKVauXMm6desIDw93HDtu3Dhef/11vv/+e2bNmuXZGypEXaj7gUlCXPj27dun3XXXXVp8fLxmNBq1kJAQbeDAgdrcuXO1/Px8x3FlDWt++OGHtaZNm2oBAQHawIEDtfXr15ca2vvWW29pgwYN0po0aaKZTCatTZs22qOPPqplZGRomqZpBQUF2qOPPqr16NFDCwkJ0YKCgrQePXqUGkZ5PsOaNU3Tdu7cqd14441aeHi4ZjabtQ4dOmhPP/204/5z585pEyZM0CIiIrTg4GBt2LBh2p49e0q9bk3TtAULFmitW7fWDAZDlSaOs5/XaDRq3bp1K1U214njSqLEEOGSJk+erAHagQMHyj3mmWee0QDtjz/+0DRNDSV+8skntVatWmn+/v5aTEyMdtNNN7mdo7i4WHv55Ze1jh07Oibzu+aaa9wm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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: mean=98.608 std=0.098, n=5\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAj4AAAGdCAYAAAASUnlxAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAp+ElEQVR4nO3df1BU56H/8Q9gYFEBoyi4XCy4jVnlGlBUrjbjvVYmKInlIp3RFq8EHWJijKPM6EACxB9hVpOGYtErbSb2Mv64aka/3G+uc0kTek3DwBcbUDPT+iPRRBMUUDO6/ggou3z/yPQ0WzB1jS3K837N7KR79jlnn7Mzzb7z7Fk2oLu7u1sAAAAGCOzrCQAAAPy9ED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjDGgrydwP/F6vTp37pzCwsIUEBDQ19MBAAB3oLu7W1evXpXdbldg4Lev6RA+33Du3DnFxsb29TQAAMBd+Pzzz/UP//AP3zqG8PmGsLAwSV+/cOHh4X08GwAAcCfcbrdiY2Ot9/FvQ/h8w58+3goPDyd8AAB4wNzJZSpc3AwAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAYxA+AADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAGP1IK4L5248YNHT9+/Dsf56uvvtJnn32muLg4hYaG3oOZSU6nUwMHDrwnxwLw90H4ALivHT9+XMnJyX09jV41NTVp4sSJfT0NAH4gfADc15xOp5qamr7zcY4dO6YFCxZox44dGjt27D2Y2ddzA/BgIXwA3NcGDhx4T1dVxo4dyyoNYDAubgYAAMYgfAAAgDEIHwAAYAzCBwAAGIPwAQAAxiB8AACAMQgfAABgDMIHAAAYg/ABAADGIHwAAIAxCB8AAGAMwgcAABiD8AEAAMYgfAAAgDHuKny2bNmiuLg42Ww2paSk6NChQ7cde+vWLa1bt04Oh0M2m02JiYmqqanxGePxeFRcXKz4+HiFhobK4XBo/fr16u7utsYEBAT0envttdesMXFxcT0e37Bhw92cIgAA6IcG+LvDnj17lJ+fr8rKSqWkpKi8vFxpaWk6ceKERowY0WN8UVGRduzYoTfeeENOp1PvvPOOMjMzVV9frwkTJkiSNm7cqK1bt6qqqkoJCQn68MMPlZubq4iICC1fvlySdP78eZ/j/s///I8WL16srKwsn+3r1q1TXl6edT8sLMzfUwQAAP2U3ys+ZWVlysvLU25ursaNG6fKykoNHDhQ27Zt63X89u3b9eKLLyo9PV2jR4/Wc889p/T0dL3++uvWmPr6emVkZOjJJ59UXFycfvzjH+uJJ57wWUmKjo72uf3Xf/2XZsyYodGjR/s8X1hYmM+4QYMG+XuKAACgn/IrfG7evKmmpialpqb++QCBgUpNTVVDQ0Ov+3R2dspms/lsCw0NVV1dnXV/2rRpqq2t1cmTJyVJR48eVV1dnWbPnt3rMdva2nTgwAEtXry4x2MbNmzQsGHDNGHCBL322mvq6uq67fl0dnbK7Xb73AAAQP/l10ddFy9elMfjUVRUlM/2qKgoHT9+vNd90tLSVFZWpunTp8vhcKi2tlb79++Xx+OxxhQUFMjtdsvpdCooKEgej0elpaXKzs7u9ZhVVVUKCwvT3LlzfbYvX75cEydO1NChQ1VfX6/CwkKdP39eZWVlvR7H5XJp7dq1/rwEAADgAeb3NT7+2rRpk/Ly8uR0OhUQECCHw6Hc3Fyfj8b27t2rnTt3ateuXUpISNCRI0e0YsUK2e125eTk9Djmtm3blJ2d3WMlKT8/3/rfjz32mIKDg7VkyRK5XC6FhIT0OE5hYaHPPm63W7GxsffitAEAwH3Ir/CJjIxUUFCQ2trafLa3tbUpOjq6132GDx+u6upqdXR06NKlS7Lb7SooKPC5NmfVqlUqKCjQ/PnzJUnjx4/XmTNn5HK5eoTPBx98oBMnTmjPnj1/db4pKSnq6urSZ599pkcffbTH4yEhIb0GEQAA6J/8usYnODhYycnJqq2ttbZ5vV7V1tZq6tSp37qvzWZTTEyMurq6tG/fPmVkZFiP3bhxQ4GBvlMJCgqS1+vtcZw333xTycnJSkxM/KvzPXLkiAIDA3v9thkAADCP3x915efnKycnR5MmTdKUKVNUXl6u69evKzc3V5K0cOFCxcTEyOVySZIaGxvV0tKipKQktbS0aM2aNfJ6vVq9erV1zDlz5qi0tFSjRo1SQkKCDh8+rLKyMi1atMjnud1ut9566y2fb4T9SUNDgxobGzVjxgyFhYWpoaFBK1eu1IIFC/Twww/7e5oAAKAf8jt85s2bpwsXLqikpEStra1KSkpSTU2NdcHz2bNnfVZvOjo6VFRUpNOnT2vw4MFKT0/X9u3bNWTIEGtMRUWFiouLtXTpUrW3t8tut2vJkiUqKSnxee7du3eru7tbP/nJT3rMKyQkRLt379aaNWvU2dmp+Ph4rVy50ucaHgAAYLaA7m/+eWTDud1uRURE6MqVKwoPD+/r6QC4h5qbm5WcnKympiZNnDixr6cD4B7y5/2b3+oCAADGIHwAAIAxCB8AAGAMwgcAABiD8AEAAMYgfAAAgDEIHwAAYAzCBwAAGIPwAQAAxiB8AACAMQgfAABgDMIHAAAYg/ABAADGIHwAAIAxCB8AAGAMwgcAABiD8AEAAMYgfAAAgDEIHwAAYAzCBwAAGIPwAQAAxiB8AACAMQgfAABgDMIHAAAYg/ABAADGIHwAAIAxCB8AAGAMwgcAABiD8AEAAMYgfAAAgDEIHwAAYAzCBwAAGIPwAQAAxiB8AACAMQgfAABgDMIHAAAYg/ABAADGIHwAAIAxCB8AAGAMwgcAABiD8AEAAMYgfAAAgDEIHwAAYAzCBwAAGIPwAQAAxiB8AACAMQgfAABgDMIHAAAYg/ABAADGIHwAAIAxCB8AAGAMwgcAABjjrsJny5YtiouLk81mU0pKig4dOnTbsbdu3dK6devkcDhks9mUmJiompoanzEej0fFxcWKj49XaGioHA6H1q9fr+7ubmtMQEBAr7fXXnvNGvPll18qOztb4eHhGjJkiBYvXqxr167dzSkCAIB+yO/w2bNnj/Lz8/Xyyy+rublZiYmJSktLU3t7e6/ji4qK9Mtf/lIVFRX64x//qGeffVaZmZk6fPiwNWbjxo3aunWrNm/erGPHjmnjxo169dVXVVFRYY05f/68z23btm0KCAhQVlaWNSY7O1t/+MMf9O677+q///u/9bvf/U7PPPOMv6cIAAD6qYDuby6r3IGUlBRNnjxZmzdvliR5vV7FxsbqhRdeUEFBQY/xdrtdL730kp5//nlrW1ZWlkJDQ7Vjxw5J0lNPPaWoqCi9+eabtx3zl/71X/9VV69eVW1trSTp2LFjGjdunH7/+99r0qRJkqSamhqlp6friy++kN1u/6vn5na7FRERoStXrig8PPwOXxEAD4Lm5mYlJyerqalJEydO7OvpALiH/Hn/9mvF5+bNm2pqalJqauqfDxAYqNTUVDU0NPS6T2dnp2w2m8+20NBQ1dXVWfenTZum2tpanTx5UpJ09OhR1dXVafbs2b0es62tTQcOHNDixYutbQ0NDRoyZIgVPZKUmpqqwMBANTY23nZubrfb5wYAAPqvAf4Mvnjxojwej6Kiony2R0VF6fjx473uk5aWprKyMk2fPl0Oh0O1tbXav3+/PB6PNaagoEBut1tOp1NBQUHyeDwqLS1VdnZ2r8esqqpSWFiY5s6da21rbW3ViBEjfE9uwAANHTpUra2tvR7H5XJp7dq1d3TuAADgwfc3/1bXpk2b9Mgjj8jpdCo4OFjLli1Tbm6uAgP//NR79+7Vzp07tWvXLjU3N6uqqko/+9nPVFVV1esxt23bpuzs7B4rSf4qLCzUlStXrNvnn3/+nY4HAADub36t+ERGRiooKEhtbW0+29va2hQdHd3rPsOHD1d1dbU6Ojp06dIl2e12FRQUaPTo0daYVatWqaCgQPPnz5ckjR8/XmfOnJHL5VJOTo7P8T744AOdOHFCe/bs8dkeHR3d4wLrrq4uffnll7edW0hIiEJCQu7s5AEAwAPPrxWf4OBgJScnWxcUS19f3FxbW6upU6d+6742m00xMTHq6urSvn37lJGRYT1248YNnxUgSQoKCpLX6+1xnDfffFPJyclKTEz02T516lRdvnxZTU1N1rbf/va38nq9SklJ8ec0AQBAP+XXio8k5efnKycnR5MmTdKUKVNUXl6u69evKzc3V5K0cOFCxcTEyOVySZIaGxvV0tKipKQktbS0aM2aNfJ6vVq9erV1zDlz5qi0tFSjRo1SQkKCDh8+rLKyMi1atMjnud1ut9566y29/vrrPeY1duxYzZo1S3l5eaqsrNStW7e0bNkyzZ8//46+0QUAAPo/v8Nn3rx5unDhgkpKStTa2qqkpCTV1NRYFzyfPXvWZ/Wmo6NDRUVFOn36tAYPHqz09HRt375dQ4YMscZUVFSouLhYS5cuVXt7u+x2u5YsWaKSkhKf5969e7e6u7v1k5/8pNe57dy5U8uWLdPMmTMVGBiorKws/eIXv/D3FAEAQD/l99/x6c/4Oz5A/8Xf8QH6r7/Z3/EBAAB4kBE+AADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAYxA+AADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjDGgrycAoH/6+OOPdfXq1b6ehuXYsWM+/7xfhIWF6ZFHHunraQDGIHwA3HMff/yxxowZ09fT6NWCBQv6ego9nDx5kvgB/k4IHwD33J9Wenbs2KGxY8f28Wy+9tVXX+mzzz5TXFycQkND+3o6kr5efVqwYMF9tTIG9HeED4C/mbFjx2rixIl9PQ3LD37wg76eAoA+xsXNAADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAYxA+AADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAY9xV+GzZskVxcXGy2WxKSUnRoUOHbjv21q1bWrdunRwOh2w2mxITE1VTU+MzxuPxqLi4WPHx8QoNDZXD4dD69evV3d3tM+7YsWP60Y9+pIiICA0aNEiTJ0/W2bNnrcf/5V/+RQEBAT63Z5999m5OEQAA9EMD/N1hz549ys/PV2VlpVJSUlReXq60tDSdOHFCI0aM6DG+qKhIO3bs0BtvvCGn06l33nlHmZmZqq+v14QJEyRJGzdu1NatW1VVVaWEhAR9+OGHys3NVUREhJYvXy5JOnXqlB5//HEtXrxYa9euVXh4uP7whz/IZrP5PF9eXp7WrVtn3R84cKC/pwgAAPopv8OnrKxMeXl5ys3NlSRVVlbqwIED2rZtmwoKCnqM3759u1566SWlp6dLkp577jm99957ev3117Vjxw5JUn19vTIyMvTkk09KkuLi4vSf//mfPitJfzrGq6++am1zOBw9nm/gwIGKjo7297QAAIAB/Pqo6+bNm2pqalJqauqfDxAYqNTUVDU0NPS6T2dnZ49VmdDQUNXV1Vn3p02bptraWp08eVKSdPToUdXV1Wn27NmSJK/XqwMHDmjMmDFKS0vTiBEjlJKSourq6h7Pt3PnTkVGRuof//EfVVhYqBs3btz2fDo7O+V2u31uAACg//IrfC5evCiPx6OoqCif7VFRUWptbe11n7S0NJWVlenjjz+W1+vVu+++q/379+v8+fPWmIKCAs2fP19Op1MPPfSQJkyYoBUrVig7O1uS1N7ermvXrmnDhg2aNWuWfvOb3ygzM1Nz587V+++/bx3npz/9qXbs2KH//d//VWFhobZv364FCxbc9nxcLpciIiKsW2xsrD8vBwAAeMD4/VGXvzZt2qS8vDw5nU4FBATI4XAoNzdX27Zts8bs3btXO3fu1K5du5SQkKAjR45oxYoVstvtysnJkdfrlSRlZGRo5cqVkqSkpCTV19ersrJS//zP/yxJeuaZZ6xjjh8/XiNHjtTMmTN16tSpXj8WKywsVH5+vnXf7XYTPwAA9GN+rfhERkYqKChIbW1tPtvb2tpue13N8OHDVV1drevXr+vMmTM6fvy4Bg8erNGjR1tjVq1aZa36jB8/Xv/2b/+mlStXyuVyWc87YMAAjRs3zufYY8eO9flW119KSUmRJH3yySe9Ph4SEqLw8HCfGwAA6L/8Cp/g4GAlJyertrbW2ub1elVbW6upU6d+6742m00xMTHq6urSvn37lJGRYT1248YNBQb6TiUoKMha6QkODtbkyZN14sQJnzEnT57U9773vds+55EjRyRJI0eOvKPzAwAA/ZvfH3Xl5+crJydHkyZN0pQpU1ReXq7r169b3/JauHChYmJirNWaxsZGtbS0KCkpSS0tLVqzZo28Xq9Wr15tHXPOnDkqLS3VqFGjlJCQoMOHD6usrEyLFi2yxqxatUrz5s3T9OnTNWPGDNXU1Ojtt9/WwYMHJX39dfddu3YpPT1dw4YN00cffaSVK1dq+vTpeuyxx77LawQAAPoJv8Nn3rx5unDhgkpKStTa2qqkpCTV1NRYFzyfPXvWZ/Wmo6NDRUVFOn36tAYPHqz09HRt375dQ4YMscZUVFSouLhYS5cuVXt7u+x2u5YsWaKSkhJrTGZmpiorK+VyubR8+XI9+uij2rdvnx5//HFJX68Kvffee1aIxcbGKisrS0VFRXf72gAAgH4moPsv/zyywdxutyIiInTlyhWu9wG+g+bmZiUnJ6upqUkTJ07s6+nct3idgHvDn/dvfqsLAAAYg/ABAADGIHwAAIAxCB8AAGAMwgcAABiD8AEAAMYgfAAAgDEIHwAAYAzCBwAAGIPwAQAAxiB8AACAMQgfAABgDMIHAAAYg/ABAADGIHwAAIAxCB8AAGAMwgcAABiD8AEAAMYgfAAAgDEIHwAAYAzCBwAAGIPwAQAAxiB8AACAMQgfAABgDMIHAAAYg/ABAADGIHwAAIAxCB8AAGAMwgcAABiD8AEAAMYgfAAAgDEIHwAAYAzCBwAAGIPwAQAAxiB8AACAMQgfAABgDMIHAAAYg/ABAADGIHwAAIAxCB8AAGAMwgcAABiD8AEAAMYgfAAAgDEIHwAAYAzCBwAAGIPwAQAAxiB8AACAMQgfAABgDMIHAAAYg/ABAADGIHwAAIAxCB8AAGCMuwqfLVu2KC4uTjabTSkpKTp06NBtx966dUvr1q2Tw+GQzWZTYmKiampqfMZ4PB4VFxcrPj5eoaGhcjgcWr9+vbq7u33GHTt2TD/60Y8UERGhQYMGafLkyTp79qz1eEdHh55//nkNGzZMgwcPVlZWltra2u7mFAEAQD/kd/js2bNH+fn5evnll9Xc3KzExESlpaWpvb291/FFRUX65S9/qYqKCv3xj3/Us88+q8zMTB0+fNgas3HjRm3dulWbN2/WsWPHtHHjRr366quqqKiwxpw6dUqPP/64nE6nDh48qI8++kjFxcWy2WzWmJUrV+rtt9/WW2+9pffff1/nzp3T3Llz/T1FAADQX3X7acqUKd3PP/+8dd/j8XTb7fZul8vV6/iRI0d2b9682Wfb3Llzu7Ozs637Tz75ZPeiRYu+dcy8efO6FyxYcNt5Xb58ufuhhx7qfuutt6xtx44d65bU3dDQcEfnduXKlW5J3VeuXLmj8QB619TU1C2pu6mpqa+ncl/jdQLuDX/ev/1a8bl586aampqUmppqbQsMDFRqaqoaGhp63aezs9NnVUaSQkNDVVdXZ92fNm2aamtrdfLkSUnS0aNHVVdXp9mzZ0uSvF6vDhw4oDFjxigtLU0jRoxQSkqKqqurrWM0NTXp1q1bPnNzOp0aNWrUt87N7Xb73AAAQP/lV/hcvHhRHo9HUVFRPtujoqLU2tra6z5paWkqKyvTxx9/LK/Xq3fffVf79+/X+fPnrTEFBQWaP3++nE6nHnroIU2YMEErVqxQdna2JKm9vV3Xrl3Thg0bNGvWLP3mN79RZmam5s6dq/fff1+S1NraquDgYA0ZMuSO5+ZyuRQREWHdYmNj/Xk5AADAA+Zv/q2uTZs26ZFHHpHT6VRwcLCWLVum3NxcBQb++an37t2rnTt3ateuXWpublZVVZV+9rOfqaqqStLXKz6SlJGRoZUrVyopKUkFBQV66qmnVFlZeddzKyws1JUrV6zb559//t1OFgAA3NcG+DM4MjJSQUFBPb4p1dbWpujo6F73GT58uKqrq9XR0aFLly7JbreroKBAo0ePtsasWrXKWvWRpPHjx+vMmTNyuVzKyclRZGSkBgwYoHHjxvkce+zYsdZHZtHR0bp586YuX77ss+rzbXMLCQlRSEiIPy8BAAB4gPm14hMcHKzk5GTV1tZa27xer2prazV16tRv3ddmsykmJkZdXV3at2+fMjIyrMdu3LjhswIkSUFBQdZKT3BwsCZPnqwTJ074jDl58qS+973vSZKSk5P10EMP+cztxIkTOnv27F+dGwAAMINfKz6SlJ+fr5ycHE2aNElTpkxReXm5rl+/rtzcXEnSwoULFRMTI5fLJUlqbGxUS0uLkpKS1NLSojVr1sjr9Wr16tXWMefMmaPS0lKNGjVKCQkJOnz4sMrKyrRo0SJrzKpVqzRv3jxNnz5dM2bMUE1Njd5++20dPHhQkhQREaHFixcrPz9fQ4cOVXh4uF544QVNnTpV//RP//RdXiMAANBP+B0+8+bN04ULF1RSUqLW1lYlJSWppqbGuuD57NmzPqs3HR0dKioq0unTpzV48GClp6dr+/btPh9HVVRUqLi4WEuXLlV7e7vsdruWLFmikpISa0xmZqYqKyvlcrm0fPlyPfroo9q3b58ef/xxa8zPf/5zBQYGKisrS52dnUpLS9O///u/383rAgAA+qGA7u6/+PPIBnO73YqIiNCVK1cUHh7e19MBHljNzc1KTk5WU1OTJk6c2NfTuW/xOgH3hj/v3/xWFwAAMAbhAwAAjEH4AAAAYxA+AADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAYxA+AADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAYxA+AADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAYxA+AADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAYxA+AADAGHcVPlu2bFFcXJxsNptSUlJ06NCh2469deuW1q1bJ4fDIZvNpsTERNXU1PiM8Xg8Ki4uVnx8vEJDQ+VwOLR+/Xp1d3dbY55++mkFBAT43GbNmuVznLi4uB5jNmzYcDenCAAA+qEB/u6wZ88e5efnq7KyUikpKSovL1daWppOnDihESNG9BhfVFSkHTt26I033pDT6dQ777yjzMxM1dfXa8KECZKkjRs3auvWraqqqlJCQoI+/PBD5ebmKiIiQsuXL7eONWvWLP3617+27oeEhPR4vnXr1ikvL8+6HxYW5u8pAgCAfsrvFZ+ysjLl5eUpNzdX48aNU2VlpQYOHKht27b1On779u168cUXlZ6ertGjR+u5555Tenq6Xn/9dWtMfX29MjIy9OSTTyouLk4//vGP9cQTT/RYSQoJCVF0dLR1e/jhh3s8X1hYmM+YQYMG+XuKAACgn/JrxefmzZtqampSYWGhtS0wMFCpqalqaGjodZ/Ozk7ZbDafbaGhoaqrq7PuT5s2Tb/61a908uRJjRkzRkePHlVdXZ3Kysp89jt48KBGjBihhx9+WD/84Q/1yiuvaNiwYT5jNmzYoPXr12vUqFH66U9/qpUrV2rAgN5Ps7OzU52dndZ9t9t9Zy8EgG/VcfVLTYgO1Jn/938VevlkX09H0tf/fz937pzsdnuvq8V9ofXTTzUhOlABXR19PRXAGH6Fz8WLF+XxeBQVFeWzPSoqSsePH+91n7S0NJWVlWn69OlyOByqra3V/v375fF4rDEFBQVyu91yOp0KCgqSx+NRaWmpsrOzrTGzZs3S3LlzFR8fr1OnTunFF1/U7Nmz1dDQoKCgIEnS8uXLNXHiRA0dOlT19fUqLCzU+fPnewTUn7hcLq1du9aflwDAHWj7Q52alwyW2n8utff1bP4sSZI+7+NJfMNYSelLButs96W+ngpgjIDub15B/FecO3dOMTExqq+v19SpU63tq1ev1vvvv6/GxsYe+1y4cEF5eXl6++23FRAQIIfDodTUVG3btk1fffWVJGn37t1atWqVXnvtNSUkJOjIkSNasWKFysrKlJOT0+tcTp8+LYfDoffee08zZ87sdcy2bdu0ZMkSXbt2rdf/wuttxSc2NlZXrlxReHj4nb4sAP7CxfOf64P/86b1JYj7waeffqqioiK98sorio+P7+vpWAYNGqRRE2ZKwQP7eirAA8vtdisiIuKO3r/9WvGJjIxUUFCQ2trafLa3tbUpOjq6132GDx+u6upqdXR06NKlS7Lb7SooKNDo0aOtMatWrVJBQYHmz58vSRo/frzOnDkjl8t12/AZPXq0IiMj9cknn9w2fFJSUtTV1aXPPvtMjz76aI/HQ0JC7pslb6A/iRwZq8yla/p6Gj6+am7W4dYXFT0hTWMnTuzr6QDoI35d3BwcHKzk5GTV1tZa27xer2pra31WgHpjs9kUExOjrq4u7du3TxkZGdZjN27cUGCg71SCgoLk9Xpve7wvvvhCly5d0siRI2875siRIwoMDOz122YAAMA8fn+dPT8/Xzk5OZo0aZKmTJmi8vJyXb9+Xbm5uZKkhQsXKiYmRi6XS5LU2NiolpYWJSUlqaWlRWvWrJHX69Xq1autY86ZM0elpaUaNWqUEhISdPjwYZWVlWnRokWSpGvXrmnt2rXKyspSdHS0Tp06pdWrV+v73/++0tLSJEkNDQ1qbGzUjBkzFBYWpoaGBq1cuVILFizo9dtfAADAPH6Hz7x583ThwgWVlJSotbVVSUlJqqmpsS54Pnv2rM/qTUdHh4qKinT69GkNHjxY6enp2r59u4YMGWKNqaioUHFxsZYuXar29nbZ7XYtWbJEJSUlkr5e/fnoo49UVVWly5cvy26364knntD69eutj6pCQkK0e/durVmzRp2dnYqPj9fKlSuVn5//XV4fAADQj/h1cXN/58/FUQAeLM3NzUpOTlZTU5Mmco0P0K/48/7Nb3UBAABjED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAYxA+AADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAYxA+AADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAYxA+AADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAYxA+AADAGIQPAAAwBuEDAACMQfgAAABjED4AAMAYhA8AADAG4QMAAIxxV+GzZcsWxcXFyWazKSUlRYcOHbrt2Fu3bmndunVyOByy2WxKTExUTU2NzxiPx6Pi4mLFx8crNDRUDodD69evV3d3tzXm6aefVkBAgM9t1qxZPsf58ssvlZ2drfDwcA0ZMkSLFy/WtWvX7uYUAQBAPzTA3x327Nmj/Px8VVZWKiUlReXl5UpLS9OJEyc0YsSIHuOLioq0Y8cOvfHGG3I6nXrnnXeUmZmp+vp6TZgwQZK0ceNGbd26VVVVVUpISNCHH36o3NxcRUREaPny5daxZs2apV//+tfW/ZCQEJ/nys7O1vnz5/Xuu+/q1q1bys3N1TPPPKNdu3b5e5oAAKAfCuj+5rLKHUhJSdHkyZO1efNmSZLX61VsbKxeeOEFFRQU9Bhvt9v10ksv6fnnn7e2ZWVlKTQ0VDt27JAkPfXUU4qKitKbb7552zFPP/20Ll++rOrq6l7ndezYMY0bN06///3vNWnSJElSTU2N0tPT9cUXX8hut//Vc3O73YqIiNCVK1cUHh5+Zy8IgAdCc3OzkpOT1dTUpIkTJ/b1dADcQ/68f/u14nPz5k01NTWpsLDQ2hYYGKjU1FQ1NDT0uk9nZ6dsNpvPttDQUNXV1Vn3p02bpl/96lc6efKkxowZo6NHj6qurk5lZWU++x08eFAjRozQww8/rB/+8Id65ZVXNGzYMElSQ0ODhgwZYkWPJKWmpiowMFCNjY3KzMzsdW6dnZ3Wfbfb7cerAeDv4caNGzp+/Ph3Ps6xY8d8/nkvOJ1ODRw48J4dD8Dfnl/hc/HiRXk8HkVFRflsj4qKuu2/mNLS0lRWVqbp06fL4XCotrZW+/fvl8fjscYUFBTI7XbL6XQqKChIHo9HpaWlys7OtsbMmjVLc+fOVXx8vE6dOqUXX3xRs2fPVkNDg4KCgtTa2trjo7YBAwZo6NCham1t7XVuLpdLa9eu9eclAPB3dvz4cSUnJ9+z4y1YsOCeHYvVI+DB4/c1Pv7atGmT8vLy5HQ6FRAQIIfDodzcXG3bts0as3fvXu3cuVO7du1SQkKCjhw5ohUrVshutysnJ0eSNH/+fGv8+PHj9dhjj8nhcOjgwYOaOXPmXc2tsLBQ+fn51n23263Y2Ni7PFMAfwtOp1NNTU3f+ThfffWVPvvsM8XFxSk0NPQezOzruQF4sPgVPpGRkQoKClJbW5vP9ra2NkVHR/e6z/Dhw1VdXa2Ojg5dunRJdrtdBQUFGj16tDVm1apVKigosOJm/PjxOnPmjFwulxU+f2n06NGKjIzUJ598opkzZyo6Olrt7e0+Y7q6uvTll1/edm4hISE9LpAGcH8ZOHDgPVtV+cEPfnBPjgPgweXX19mDg4OVnJys2tpaa5vX61Vtba2mTp36rfvabDbFxMSoq6tL+/btU0ZGhvXYjRs3FBjoO5WgoCB5vd7bHu+LL77QpUuXNHLkSEnS1KlTdfnyZZ//Mvztb38rr9erlJQUf04TAAD0U35/1JWfn6+cnBxNmjRJU6ZMUXl5ua5fv67c3FxJ0sKFCxUTEyOXyyVJamxsVEtLi5KSktTS0qI1a9bI6/Vq9erV1jHnzJmj0tJSjRo1SgkJCTp8+LDKysq0aNEiSdK1a9e0du1aZWVlKTo6WqdOndLq1av1/e9/X2lpaZKksWPHatasWcrLy1NlZaVu3bqlZcuWaf78+Xf0jS4AAND/+R0+8+bN04ULF1RSUqLW1lYlJSWppqbGuuD57NmzPqs3HR0dKioq0unTpzV48GClp6dr+/btGjJkiDWmoqJCxcXFWrp0qdrb22W327VkyRKVlJRI+nr156OPPlJVVZUuX74su92uJ554QuvXr/f5qGrnzp1atmyZZs6cqcDAQGVlZekXv/jF3b42AACgn/H77/j0Z/wdHwAAHjz+vH/zW10AAMAYhA8AADAG4QMAAIxB+AAAAGMQPgAAwBiEDwAAMAbhAwAAjEH4AAAAYxA+AADAGH7/ZEV/9qc/Yu12u/t4JgAA4E796X37Tn6MgvD5hqtXr0qSYmNj+3gmAADAX1evXlVERMS3juG3ur7B6/Xq3LlzCgsLU0BAQF9PB8A95Ha7FRsbq88//5zf4gP6me7ubl29elV2u93nh9J7Q/gAMAI/QgxA4uJmAABgEMIHAAAYg/ABYISQkBC9/PLLCgkJ6eupAOhDXOMDAACMwYoPAAAwBuEDAACMQfgAAABjED4AAMAYhA+Afu13v/ud5syZI7vdroCAAFVXV/f1lAD0IcIHQL92/fp1JSYmasuWLX09FQD3AX6kFEC/Nnv2bM2ePbuvpwHgPsGKDwAAMAbhAwAAjEH4AAAAYxA+AADAGIQPAAAwBt/qAtCvXbt2TZ988ol1/9NPP9WRI0c0dOhQjRo1qg9nBqAv8OvsAPq1gwcPasaMGT225+Tk6D/+4z/+/hMC0KcIHwAAYAyu8QEAAMYgfAAAgDEIHwAAYAzCBwAAGIPwAQAAxiB8AACAMQgfAABgDMIHAAAYg/ABAADGIHwAAIAxCB8AAGAMwgcAABjj/wMzEoltWe+InQAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "efl4Yi8HGyye"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}