{
  "cells": [
    {
      "cell_type": "markdown",
      "id": "d8097d6e",
      "metadata": {
        "id": "d8097d6e"
      },
      "source": [
        "# Project Overview\n",
        "\n",
        "This project investigates the social and behavioral factors associated with Actual Usage Behavior (AUB) in social commerce among Generation Z university students in Vietnam. Using a secondary dataset of **757 survey responses**, collected between **October 2024 and March 2025** through **voluntary online and offline channels**, the study examines how perceived ease of use, perceived usefulness, familiarity, social participation, trust, and purchase intention relate to actual social commerce engagement. The project aims to identify meaningful user patterns and insights into social commerce adoption among young consumers."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "Ma2gRi52yP09",
      "metadata": {
        "id": "Ma2gRi52yP09"
      },
      "source": [
        "## CBDATSI Phase 1\n",
        "\n",
        "Phase 1 consists of four sections: (1) dataset description, (2) data cleaning, (3) exploratory data analysis (EDA), and (4) research question formulation."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "8b5e486f",
      "metadata": {
        "id": "8b5e486f"
      },
      "source": [
        "### [1] Dataset Description\n",
        "\n",
        "The dataset was collected through an online survey conducted by researchers from the Industrial University of Ho Chi Minh City and Thai Nguyen University of Economics & Business Administration. Participation was voluntary and limited to respondents aged 18–27 years.\n",
        "\n",
        "Because only Generation Z respondents were included, the findings cannot be generalized to other age groups.\n",
        "\n",
        "The dataset consists of a single CSV file containing all survey responses, while the questionnaire and codebook provide supporting documentation. Each row represents one respondent, and each column represents either a demographic variable or a survey item measuring a social commerce construct."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "id": "f444af3c",
      "metadata": {
        "id": "f444af3c"
      },
      "outputs": [],
      "source": [
        "# Code explanation: Import the libraries needed to be used.\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import scikit_posthocs as sp\n",
        "\n",
        "from scipy import stats\n",
        "from scipy.stats import shapiro\n",
        "from scipy.stats import kruskal\n",
        "\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.cluster import KMeans\n",
        "from sklearn.metrics import silhouette_score\n",
        "\n",
        "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
        "\n",
        "import os\n",
        "import sys\n",
        "\n",
        "sys.path.append(os.path.abspath(\"..\"))\n",
        "\n",
        "# sets the theme of the charts\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "\n",
        "%matplotlib inline"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "id": "ff66917d",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ff66917d",
        "outputId": "668c0cbb-8407-47e8-9eb3-ca38f4e49291"
      },
      "outputs": [],
      "source": [
        "# Get the folder where this notebook is located\n",
        "BASE_DIR = os.getcwd()\n",
        "\n",
        "# Import preprocessing wrapper functions\n",
        "from preprocessing import (\n",
        "    load_dataset,\n",
        "    inspect_dataset,\n",
        "    validate_columns,\n",
        "    validate_dtypes,\n",
        "    drop_columns,\n",
        "    check_missing_values,\n",
        "    find_duplicates,\n",
        "    validate_unique_values,\n",
        "    compute_composite_score,\n",
        ")\n",
        "\n",
        "# Load the dataset from the same folder as the notebook\n",
        "scommerce_df = pd.read_csv(\n",
        "    os.path.join(BASE_DIR, \"S-COMMERCE_GenZ_UniversityStudent_757_DIB copy.csv\")\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "adZN42LMb-H2",
      "metadata": {
        "id": "adZN42LMb-H2"
      },
      "source": [
        "The dataset contains **757 observations** and **31 variables**."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "id": "kcGLbEg5cAkS",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kcGLbEg5cAkS",
        "outputId": "a4af2af2-76ab-4205-ceb8-5855470d4dc2"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Rows (observations): 757\n",
            "Columns (variables): 31\n"
          ]
        }
      ],
      "source": [
        "rows, columns = inspect_dataset(scommerce_df)\n",
        "\n",
        "print(f\"Rows (observations): {rows}\")\n",
        "print(f\"Columns (variables): {columns}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "VfLHbbOqerVK",
      "metadata": {
        "id": "VfLHbbOqerVK"
      },
      "source": [
        "The variables include demographic information (Gender, Income, Area, and Frequently) and survey items measuring Perceived Usefulness (PU), Perceived Ease of Use (PEU), Familiarity with Social Commerce (FSC), Social Presence (SP), Trust in Platform (TP), Interaction Behavior (IB), and Actual Usage Behavior (AUB)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "id": "P8GJLRLqetWq",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "P8GJLRLqetWq",
        "outputId": "7efb4140-3b98-4803-95c3-4e60f6ac131b"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "['Gender',\n",
              " 'Job',\n",
              " 'Income',\n",
              " 'Area',\n",
              " 'Frequently',\n",
              " 'PU1',\n",
              " 'PU2',\n",
              " 'PU3',\n",
              " 'PU4',\n",
              " 'PEU1',\n",
              " 'PEU2',\n",
              " 'PEU3',\n",
              " 'PEU4',\n",
              " 'FSC1',\n",
              " 'FSC2',\n",
              " 'FSC3',\n",
              " 'SP1',\n",
              " 'SP2',\n",
              " 'SP3',\n",
              " 'SP4',\n",
              " 'TP1',\n",
              " 'TP2',\n",
              " 'TP3',\n",
              " 'IB1',\n",
              " 'IB2',\n",
              " 'IB3',\n",
              " 'IB4',\n",
              " 'AUB1',\n",
              " 'AUB2',\n",
              " 'AUB3',\n",
              " 'AUB4']"
            ]
          },
          "execution_count": 4,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "validate_columns(scommerce_df)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "BZlHEv4ae7wu",
      "metadata": {
        "id": "BZlHEv4ae7wu"
      },
      "source": [
        "The **Gender** variable has three categories:\n",
        "- 1 – Male\n",
        "- 2 – Female\n",
        "- 3 – Different"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "id": "jwM0uyQJe2Wa",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 209
        },
        "id": "jwM0uyQJe2Wa",
        "outputId": "3f4c948c-e56b-4bf9-9548-af6499b0117f"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Gender\n",
              "1    167\n",
              "2    588\n",
              "3      2\n",
              "Name: count, dtype: int64"
            ]
          },
          "execution_count": 5,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df['Gender'].value_counts().sort_index()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "ShBqmuJFfhMl",
      "metadata": {
        "id": "ShBqmuJFfhMl"
      },
      "source": [
        "The **Income** variable has five categories:\n",
        "\n",
        "- 1 – Less than 100 USD\n",
        "- 2 – 100 to <200 USD\n",
        "- 3 – 200 to <300 USD\n",
        "- 4 – 300 to <400 USD\n",
        "- 5 – Above 400 USD"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "id": "B8e9gj31gUY-",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 272
        },
        "id": "B8e9gj31gUY-",
        "outputId": "d9ee56ae-953e-46a8-a977-f6836e823380"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Income\n",
              "1    672\n",
              "2     68\n",
              "3     12\n",
              "4      2\n",
              "5      3\n",
              "Name: count, dtype: int64"
            ]
          },
          "execution_count": 6,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df['Income'].value_counts().sort_index()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "fKXjQ_9CgZpU",
      "metadata": {
        "id": "fKXjQ_9CgZpU"
      },
      "source": [
        "The **Area** variable has three categories:\n",
        "\n",
        "- 1 – Urban\n",
        "- 2 – Suburban\n",
        "- 3 – Rural"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "id": "xyalc6gvgifq",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 209
        },
        "id": "xyalc6gvgifq",
        "outputId": "8ef98763-4ea2-47f8-8329-70a3bc805c97"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Area\n",
              "1    461\n",
              "2     74\n",
              "3    222\n",
              "Name: count, dtype: int64"
            ]
          },
          "execution_count": 7,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df['Area'].value_counts().sort_index()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "3OFVdtHHgrX9",
      "metadata": {
        "id": "3OFVdtHHgrX9"
      },
      "source": [
        "The **Frequently** variable measures social media usage frequency:\n",
        "\n",
        "- 1 – Daily\n",
        "- 2 – Weekly\n",
        "- 3 – Monthly\n",
        "- 4 – Rarely Used"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "id": "KycxA7MogsBB",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 240
        },
        "id": "KycxA7MogsBB",
        "outputId": "d79b9e4e-44fe-4c12-810e-d691c7cb8fc3"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Frequently\n",
              "1    725\n",
              "2     14\n",
              "3      7\n",
              "4     11\n",
              "Name: count, dtype: int64"
            ]
          },
          "execution_count": 8,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df['Frequently'].value_counts().sort_index()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "A_g0wlXRhDlD",
      "metadata": {
        "id": "A_g0wlXRhDlD"
      },
      "source": [
        "#### For the following constructs, they use the following scale:\n",
        "* 1 - Strongly disagree,\n",
        "* 2 - Disagree\n",
        "* 3 - Neutral\n",
        "* 4 - Agree\n",
        "* 5 - Strongly agree"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "Ny0SmQAHhlhT",
      "metadata": {
        "id": "Ny0SmQAHhlhT"
      },
      "source": [
        "#### Perceived Usefulness (PU)\n",
        "Measure the extent to which respondents believe social commerce improves their shopping effectiveness and productivity.\n",
        "* `PU1`: Using social commerce platforms improves online shopping performance.\n",
        "* `PU2`: ... increases productivity in finding products.\n",
        "* `PU3`: ... enhances effectiveness when making online purchases.\n",
        "* `PU4`: ... are useful for online transactions."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "id": "McBVjf_6lBpM",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "McBVjf_6lBpM",
        "outputId": "ca0ab8d7-83f4-42ee-b0db-dd18dfbc8761"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PU1</th>\n",
              "      <th>PU2</th>\n",
              "      <th>PU3</th>\n",
              "      <th>PU4</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PU1  PU2  PU3  PU4\n",
              "0    4    4    4    4\n",
              "1    3    3    3    3\n",
              "2    3    3    3    3\n",
              "3    5    5    5    5\n",
              "4    4    4    4    4"
            ]
          },
          "execution_count": 9,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df[['PU1','PU2','PU3','PU4']].head()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "_fAtTOGclGsa",
      "metadata": {
        "id": "_fAtTOGclGsa"
      },
      "source": [
        "\n",
        "#### Perceived Ease of Use (PEU)\n",
        "Measure how easy respondents find social commerce platforms to use.\n",
        "* `PEU1`: Interaction with social commerce platforms is clear and understandable.\n",
        "* `PEU2`: Using social commerce platforms requires little mental effort.\n",
        "* `PEU3`: It is easy to make social commerce platforms perform desired tasks.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "id": "jFIXXtb6lVqL",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "jFIXXtb6lVqL",
        "outputId": "c9393f3a-c3a8-49ef-fa95-ae6553fd1622"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PEU1</th>\n",
              "      <th>PEU2</th>\n",
              "      <th>PEU3</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PEU1  PEU2  PEU3\n",
              "0     4     4     4\n",
              "1     3     3     3\n",
              "2     3     3     3\n",
              "3     5     5     5\n",
              "4     4     4     4"
            ]
          },
          "execution_count": 10,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df[['PEU1','PEU2','PEU3']].head()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "9TcR0FN358cE",
      "metadata": {
        "id": "9TcR0FN358cE"
      },
      "source": [
        "However, the CSV file contains a variable `PEU4` that is not documented in the codebook, questionnaire, and its research paper."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "id": "7VqxwWDO8NQV",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 240
        },
        "id": "7VqxwWDO8NQV",
        "outputId": "c09cf08f-a843-4618-a4a1-669842522efb"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "0    4\n",
              "1    3\n",
              "2    3\n",
              "3    5\n",
              "4    4\n",
              "Name: PEU4, dtype: int64"
            ]
          },
          "execution_count": 11,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df['PEU4'].head()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "f4GC0A6w8Uvf",
      "metadata": {
        "id": "f4GC0A6w8Uvf"
      },
      "source": [
        "Since `PEU4` is not defined in the codebook or questionnaire, its meaning cannot be accurately determined. Therefore, the variable will be removed during data cleaning to maintain data consistency and interpretability throughout the analysis."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "jnDhUvVOlJUs",
      "metadata": {
        "id": "jnDhUvVOlJUs"
      },
      "source": [
        "#### Familiarity with Social Commerce (FSC)\n",
        "Measure respondents' familiarity with social commerce platforms.\n",
        "* `FSC1`: Familiarity gained through personal experience using social commerce platforms.\n",
        "* `FSC2`: ... reading news, blogs, and related materials.\n",
        "* `FSC3`: ... communication with others regarding social commerce.\n",
        "* `FSC4`: ... discussions, reviews, and shared experiences.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "id": "Y_vDdFuOld9U",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "Y_vDdFuOld9U",
        "outputId": "4af6c3ae-8d17-4fc7-93e5-3bfb5e74c3ed"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>FSC1</th>\n",
              "      <th>FSC2</th>\n",
              "      <th>FSC3</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   FSC1  FSC2  FSC3\n",
              "0     4     4     4\n",
              "1     3     3     3\n",
              "2     3     3     3\n",
              "3     5     5     5\n",
              "4     3     3     3"
            ]
          },
          "execution_count": 12,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df[['FSC1','FSC2','FSC3']].head()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "SAmnjtzA7Twn",
      "metadata": {
        "id": "SAmnjtzA7Twn"
      },
      "source": [
        "As seen from the code block above, FSC4 is no longer called as it does not exist in the dataset. Even though it is defined in the codebook, questionnaire, and the study, since it does not appear in the CSV file of the dataset, calling it will only cause errors. Therefore, it is disregarded. Further explanation will be provided in the Data Cleaning section."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "R3UI-SIdlLSF",
      "metadata": {
        "id": "R3UI-SIdlLSF"
      },
      "source": [
        "\n",
        "#### Social Presence (SP)\n",
        "Measure the degree of social interaction experienced through social commerce platforms.\n",
        "* `SP1`: Frequency of attending online group meetings or events through social commerce platforms.\n",
        "* `SP2`: ... socializing with friends through social commerce platforms.\n",
        "* `SP3`: ... socializing with relatives through social commerce platforms.\n",
        "* `SP4`: General level of social interaction and engagement on social commerce platforms."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "id": "CBQWn5UVltD4",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "CBQWn5UVltD4",
        "outputId": "c0c57608-205f-46c9-9067-71602133d696"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>SP1</th>\n",
              "      <th>SP2</th>\n",
              "      <th>SP3</th>\n",
              "      <th>SP4</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>2</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   SP1  SP2  SP3  SP4\n",
              "0    4    4    4    4\n",
              "1    3    3    3    3\n",
              "2    3    3    3    3\n",
              "3    5    5    5    5\n",
              "4    2    3    3    3"
            ]
          },
          "execution_count": 13,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df[['SP1','SP2','SP3', 'SP4']].head()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "J6GAtIxSlNJg",
      "metadata": {
        "id": "J6GAtIxSlNJg"
      },
      "source": [
        "\n",
        "#### Trust in Platform (TP)\n",
        "Measure respondents' trust in social commerce platforms.\n",
        "* `TP1`: Social commerce platforms are competent and effective.\n",
        "* `TP2`: ... act in users' best interests.\n",
        "* `TP3`: ... can be trusted consistently.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "id": "DrgEJr5Yl1uM",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "DrgEJr5Yl1uM",
        "outputId": "869930a2-59e0-46b1-fecb-98425828a128"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>TP1</th>\n",
              "      <th>TP2</th>\n",
              "      <th>TP3</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   TP1  TP2  TP3\n",
              "0    4    4    4\n",
              "1    3    3    3\n",
              "2    3    3    3\n",
              "3    5    5    5\n",
              "4    3    3    3"
            ]
          },
          "execution_count": 14,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df[['TP1','TP2','TP3']].head()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "auxBR2K1lPCq",
      "metadata": {
        "id": "auxBR2K1lPCq"
      },
      "source": [
        "\n",
        "#### Interaction Behavior (IB)\n",
        "Measure respondents' willingness to interact and share information with others through social commerce platforms.\n",
        "* `IB1`: Willingness to provide information to vendors.\n",
        "* `IB2`: ... share shopping experiences and suggestions with friends.\n",
        "* `IB3`: ... purchase products recommended by friends.\n",
        "* `IB4`: ... consider friends' shopping experiences when making purchases."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "id": "1OD7gJGnl9fb",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "1OD7gJGnl9fb",
        "outputId": "d3997939-665f-46bb-baba-2d3f533643d9"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>IB1</th>\n",
              "      <th>IB2</th>\n",
              "      <th>IB3</th>\n",
              "      <th>IB4</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   IB1  IB2  IB3  IB4\n",
              "0    3    3    3    3\n",
              "1    3    3    3    3\n",
              "2    3    3    3    3\n",
              "3    5    5    5    5\n",
              "4    4    4    3    4"
            ]
          },
          "execution_count": 15,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df[['IB1','IB2','IB3', 'IB4']].head()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "h3crDueccA4m",
      "metadata": {
        "id": "h3crDueccA4m"
      },
      "source": [
        "#### Actual Usage Behavior (AUB)\n",
        "Measure respondents' actual usage of social commerce platforms.\n",
        "* `AUB1`: Using social commerce platforms is enjoyable.\n",
        "* `AUB2`: Uses social commerce platforms for safe online shopping.\n",
        "* `AUB3`: Spends significant time on social commerce platforms.\n",
        "* `AUB4`: Regularly uses social commerce platforms."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "37bb299a",
      "metadata": {
        "id": "37bb299a"
      },
      "source": [
        "### [2] Data Cleaning\n",
        "\n",
        "Before analysis, the dataset was inspected for data quality issues that could affect the validity and reliability of the results. The variables were checked for missing values, duplicate records, incorrect data types, inconsistent formatting, and invalid values. Appropriate preprocessing was applied where necessary."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "gFfwyM6FrlU6",
      "metadata": {
        "id": "gFfwyM6FrlU6"
      },
      "source": [
        "#### Variable Verification\n",
        "\n",
        "Before cleaning, the dataset variables were verified against the questionnaire and accompanying study to ensure that all expected variables were correctly loaded and matched the constructs defined by the authors."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "id": "PlIAL3dltRnO",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PlIAL3dltRnO",
        "outputId": "9dfb5101-2db1-48cd-f7fc-34bf4acc2d84"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Index(['Gender', 'Job', 'Income', 'Area', 'Frequently', 'PU1', 'PU2', 'PU3',\n",
              "       'PU4', 'PEU1', 'PEU2', 'PEU3', 'PEU4', 'FSC1', 'FSC2', 'FSC3', 'SP1',\n",
              "       'SP2', 'SP3', 'SP4', 'TP1', 'TP2', 'TP3', 'IB1', 'IB2', 'IB3', 'IB4',\n",
              "       'AUB1', 'AUB2', 'AUB3', 'AUB4'],\n",
              "      dtype='str')"
            ]
          },
          "execution_count": 16,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df.columns"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "pi1cXxq1Jbob",
      "metadata": {
        "id": "pi1cXxq1Jbob"
      },
      "source": [
        "Since `PEU4` is not documented in the codebook, questionnaire, or associated research paper, its meaning cannot be verified. Therefore, the variable is removed to avoid ambiguity in the analysis."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "id": "RNVxTX89IgrA",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 252
        },
        "id": "RNVxTX89IgrA",
        "outputId": "0f18c1b4-e7bd-4037-8600-172ab9671ba1"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "(757, 30)\n"
          ]
        },
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Gender</th>\n",
              "      <th>Job</th>\n",
              "      <th>Income</th>\n",
              "      <th>Area</th>\n",
              "      <th>Frequently</th>\n",
              "      <th>PU1</th>\n",
              "      <th>PU2</th>\n",
              "      <th>PU3</th>\n",
              "      <th>PU4</th>\n",
              "      <th>PEU1</th>\n",
              "      <th>...</th>\n",
              "      <th>TP2</th>\n",
              "      <th>TP3</th>\n",
              "      <th>IB1</th>\n",
              "      <th>IB2</th>\n",
              "      <th>IB3</th>\n",
              "      <th>IB4</th>\n",
              "      <th>AUB1</th>\n",
              "      <th>AUB2</th>\n",
              "      <th>AUB3</th>\n",
              "      <th>AUB4</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>...</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "      <td>2</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>...</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 30 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "   Gender  Job  Income  Area  Frequently  PU1  PU2  PU3  PU4  PEU1  ...  TP2  \\\n",
              "0       2    1       1     1           1    4    4    4    4     4  ...    4   \n",
              "1       2    1       1     1           1    3    3    3    3     3  ...    3   \n",
              "2       2    1       1     1           3    3    3    3    3     3  ...    3   \n",
              "3       1    1       1     3           1    5    5    5    5     5  ...    5   \n",
              "4       2    1       1     3           1    4    4    4    4     4  ...    3   \n",
              "\n",
              "   TP3  IB1  IB2  IB3  IB4  AUB1  AUB2  AUB3  AUB4  \n",
              "0    4    3    3    3    3     4     4     4     4  \n",
              "1    3    3    3    3    3     3     3     3     3  \n",
              "2    3    3    3    3    3     3     4     2     4  \n",
              "3    5    5    5    5    5     5     5     5     5  \n",
              "4    3    4    4    3    4     3     3     3     3  \n",
              "\n",
              "[5 rows x 30 columns]"
            ]
          },
          "execution_count": 17,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Remove the PEU4 column\n",
        "scommerce_df = drop_columns(\n",
        "    scommerce_df,\n",
        "    [\"PEU4\"]\n",
        ")\n",
        "\n",
        "# Verify removal\n",
        "print(inspect_dataset(scommerce_df))\n",
        "scommerce_df.head()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "E5SwiFXNT7Cu",
      "metadata": {
        "id": "E5SwiFXNT7Cu"
      },
      "source": [
        "The dataset dimensions and first few rows are displayed to confirm that the `PEU4` column was successfully removed."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "id": "RW-JyjohT9U-",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RW-JyjohT9U-",
        "outputId": "16ba9a7c-b039-4d11-f794-9df6af9e1203"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Index(['Gender', 'Job', 'Income', 'Area', 'Frequently', 'PU1', 'PU2', 'PU3',\n",
              "       'PU4', 'PEU1', 'PEU2', 'PEU3', 'FSC1', 'FSC2', 'FSC3', 'SP1', 'SP2',\n",
              "       'SP3', 'SP4', 'TP1', 'TP2', 'TP3', 'IB1', 'IB2', 'IB3', 'IB4', 'AUB1',\n",
              "       'AUB2', 'AUB3', 'AUB4'],\n",
              "      dtype='str')"
            ]
          },
          "execution_count": 18,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df.columns"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "73c4696f",
      "metadata": {
        "id": "73c4696f"
      },
      "source": [
        "All expected variables were successfully loaded from the dataset. An additional variable, `Job`, is present but is not documented in the questionnaire, codebook, or accompanying research paper. Its contents are inspected to determine whether it should be retained."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "id": "8e9657bc",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8e9657bc",
        "outputId": "9559fa89-a7f4-475f-b789-27d57302e939"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Unique values in Job: [1]\n",
            "Number of unique values: 1\n"
          ]
        }
      ],
      "source": [
        "# Check the unique values in the Job column\n",
        "print(\"Unique values in Job:\", scommerce_df['Job'].unique())\n",
        "print(\"Number of unique values:\", scommerce_df['Job'].nunique())"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "EQFuPyaG8G1Z",
      "metadata": {
        "id": "EQFuPyaG8G1Z"
      },
      "source": [
        "Inspection showed that all 757 values of `Job` were identical. Since the variable has no variation and is not documented in the questionnaire or research paper, it was removed from the dataset."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "id": "c6d04760",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 252
        },
        "id": "c6d04760",
        "outputId": "3cb4f54b-f970-4e9a-e9e4-43dff00b251d"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "(757, 29)\n"
          ]
        },
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Gender</th>\n",
              "      <th>Income</th>\n",
              "      <th>Area</th>\n",
              "      <th>Frequently</th>\n",
              "      <th>PU1</th>\n",
              "      <th>PU2</th>\n",
              "      <th>PU3</th>\n",
              "      <th>PU4</th>\n",
              "      <th>PEU1</th>\n",
              "      <th>PEU2</th>\n",
              "      <th>...</th>\n",
              "      <th>TP2</th>\n",
              "      <th>TP3</th>\n",
              "      <th>IB1</th>\n",
              "      <th>IB2</th>\n",
              "      <th>IB3</th>\n",
              "      <th>IB4</th>\n",
              "      <th>AUB1</th>\n",
              "      <th>AUB2</th>\n",
              "      <th>AUB3</th>\n",
              "      <th>AUB4</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>...</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "      <td>2</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>...</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 29 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "   Gender  Income  Area  Frequently  PU1  PU2  PU3  PU4  PEU1  PEU2  ...  TP2  \\\n",
              "0       2       1     1           1    4    4    4    4     4     4  ...    4   \n",
              "1       2       1     1           1    3    3    3    3     3     3  ...    3   \n",
              "2       2       1     1           3    3    3    3    3     3     3  ...    3   \n",
              "3       1       1     3           1    5    5    5    5     5     5  ...    5   \n",
              "4       2       1     3           1    4    4    4    4     4     4  ...    3   \n",
              "\n",
              "   TP3  IB1  IB2  IB3  IB4  AUB1  AUB2  AUB3  AUB4  \n",
              "0    4    3    3    3    3     4     4     4     4  \n",
              "1    3    3    3    3    3     3     3     3     3  \n",
              "2    3    3    3    3    3     3     4     2     4  \n",
              "3    5    5    5    5    5     5     5     5     5  \n",
              "4    3    4    4    3    4     3     3     3     3  \n",
              "\n",
              "[5 rows x 29 columns]"
            ]
          },
          "execution_count": 20,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Remove the Job column\n",
        "scommerce_df = drop_columns(\n",
        "    scommerce_df,\n",
        "    [\"Job\"]\n",
        ")\n",
        "\n",
        "# Verify removal\n",
        "print(inspect_dataset(scommerce_df))\n",
        "scommerce_df.head()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "6d964c16",
      "metadata": {
        "id": "6d964c16"
      },
      "source": [
        "The cleaned dataset now contains 30 documented variables."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "Vwn_uB4FwCoI",
      "metadata": {
        "id": "Vwn_uB4FwCoI"
      },
      "source": [
        "### Missing Values\n",
        "\n",
        "The dataset was checked for missing values, as incomplete records could affect subsequent analyses."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "id": "QG1-OpP5yR0N",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 993
        },
        "collapsed": true,
        "id": "QG1-OpP5yR0N",
        "outputId": "f88c9031-148c-405f-82e5-128ba69ad398"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Gender        0\n",
              "Income        0\n",
              "Area          0\n",
              "Frequently    0\n",
              "PU1           0\n",
              "PU2           0\n",
              "PU3           0\n",
              "PU4           0\n",
              "PEU1          0\n",
              "PEU2          0\n",
              "PEU3          0\n",
              "FSC1          0\n",
              "FSC2          0\n",
              "FSC3          0\n",
              "SP1           0\n",
              "SP2           0\n",
              "SP3           0\n",
              "SP4           0\n",
              "TP1           0\n",
              "TP2           0\n",
              "TP3           0\n",
              "IB1           0\n",
              "IB2           0\n",
              "IB3           0\n",
              "IB4           0\n",
              "AUB1          0\n",
              "AUB2          0\n",
              "AUB3          0\n",
              "AUB4          0\n",
              "dtype: int64"
            ]
          },
          "execution_count": 21,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "check_missing_values(scommerce_df)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "mYeAwYQ4y-vr",
      "metadata": {
        "id": "mYeAwYQ4y-vr"
      },
      "source": [
        "No missing values were found in the dataset; therefore, all observations were retained. This is consistent with the dataset documentation, which states that incomplete responses and low-quality submissions were removed during the authors' data screening process."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "ew53MwBe0jYb",
      "metadata": {
        "id": "ew53MwBe0jYb"
      },
      "source": [
        "### Duplicate Values\n",
        "\n",
        "The dataset was checked for duplicate records to ensure that each observation was unique before analysis."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "id": "53XmtQFR0t1y",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "53XmtQFR0t1y",
        "outputId": "044942f8-e435-49dd-8669-02fbed27543b"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Gender</th>\n",
              "      <th>Income</th>\n",
              "      <th>Area</th>\n",
              "      <th>Frequently</th>\n",
              "      <th>PU1</th>\n",
              "      <th>PU2</th>\n",
              "      <th>PU3</th>\n",
              "      <th>PU4</th>\n",
              "      <th>PEU1</th>\n",
              "      <th>PEU2</th>\n",
              "      <th>...</th>\n",
              "      <th>TP2</th>\n",
              "      <th>TP3</th>\n",
              "      <th>IB1</th>\n",
              "      <th>IB2</th>\n",
              "      <th>IB3</th>\n",
              "      <th>IB4</th>\n",
              "      <th>AUB1</th>\n",
              "      <th>AUB2</th>\n",
              "      <th>AUB3</th>\n",
              "      <th>AUB4</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>12</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>14</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>20</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>21</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>24</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>711</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>...</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>718</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>719</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>...</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>724</th>\n",
              "      <td>2</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>...</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>739</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>...</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>129 rows × 29 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "     Gender  Income  Area  Frequently  PU1  PU2  PU3  PU4  PEU1  PEU2  ...  \\\n",
              "12        2       1     1           1    3    3    3    3     3     3  ...   \n",
              "14        2       1     1           1    3    3    3    3     3     3  ...   \n",
              "20        2       1     1           1    3    3    3    3     3     3  ...   \n",
              "21        2       1     1           1    3    3    3    3     3     3  ...   \n",
              "24        2       1     1           1    3    3    3    3     3     3  ...   \n",
              "..      ...     ...   ...         ...  ...  ...  ...  ...   ...   ...  ...   \n",
              "711       2       1     1           1    4    4    4    4     4     4  ...   \n",
              "718       2       1     1           1    4    4    4    4     4     4  ...   \n",
              "719       2       1     1           1    4    4    4    4     4     4  ...   \n",
              "724       2       2     1           1    4    4    4    4     4     4  ...   \n",
              "739       2       1     1           1    4    4    4    4     4     4  ...   \n",
              "\n",
              "     TP2  TP3  IB1  IB2  IB3  IB4  AUB1  AUB2  AUB3  AUB4  \n",
              "12     3    3    3    3    3    3     3     3     3     3  \n",
              "14     3    3    3    3    3    3     3     3     3     3  \n",
              "20     3    3    3    3    3    3     3     3     3     3  \n",
              "21     3    3    3    3    3    3     3     3     3     3  \n",
              "24     3    3    3    3    3    3     3     3     3     3  \n",
              "..   ...  ...  ...  ...  ...  ...   ...   ...   ...   ...  \n",
              "711    4    4    4    4    4    4     4     4     4     4  \n",
              "718    3    3    4    4    4    4     4     4     4     4  \n",
              "719    4    4    4    4    4    4     4     4     4     4  \n",
              "724    4    4    4    4    4    4     4     4     4     4  \n",
              "739    4    4    4    4    4    4     4     4     4     4  \n",
              "\n",
              "[129 rows x 29 columns]"
            ]
          },
          "execution_count": 22,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Identify duplicate rows\n",
        "duplicate_rows = find_duplicates(scommerce_df)\n",
        "\n",
        "# View duplicate rows\n",
        "duplicate_rows"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "ia24yAsJ4hMy",
      "metadata": {
        "id": "ia24yAsJ4hMy"
      },
      "source": [
        "Although duplicate records were identified, the dataset contains no unique respondent identifier. Different participants may therefore have identical responses. Since the dataset documentation states that the authors had already screened the data before publication, these records were retained."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "jcYFS-Iy5UZy",
      "metadata": {
        "id": "jcYFS-Iy5UZy"
      },
      "source": [
        "### Data Type Checks\n",
        "\n",
        "The data types of each variable were verified to ensure compatibility with subsequent preprocessing and analysis."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "id": "vFsdJMt37LFr",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 993
        },
        "id": "vFsdJMt37LFr",
        "outputId": "01ad641b-1fd2-4d03-f149-92149707938c"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Gender        int64\n",
              "Income        int64\n",
              "Area          int64\n",
              "Frequently    int64\n",
              "PU1           int64\n",
              "PU2           int64\n",
              "PU3           int64\n",
              "PU4           int64\n",
              "PEU1          int64\n",
              "PEU2          int64\n",
              "PEU3          int64\n",
              "FSC1          int64\n",
              "FSC2          int64\n",
              "FSC3          int64\n",
              "SP1           int64\n",
              "SP2           int64\n",
              "SP3           int64\n",
              "SP4           int64\n",
              "TP1           int64\n",
              "TP2           int64\n",
              "TP3           int64\n",
              "IB1           int64\n",
              "IB2           int64\n",
              "IB3           int64\n",
              "IB4           int64\n",
              "AUB1          int64\n",
              "AUB2          int64\n",
              "AUB3          int64\n",
              "AUB4          int64\n",
              "dtype: object"
            ]
          },
          "execution_count": 23,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "validate_dtypes(scommerce_df)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "GpMZccnD8OXz",
      "metadata": {
        "id": "GpMZccnD8OXz"
      },
      "source": [
        "All variables were loaded using the expected numerical data types. The demographic variables (`Gender`, `Income`, `Area`, and `Frequently`) were already numerically encoded by the dataset authors, so no data type conversion was required."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "l0pqujPg8u_C",
      "metadata": {
        "id": "l0pqujPg8u_C"
      },
      "source": [
        "### Formatting and Range Check\n",
        "\n",
        "After verifying the data types, the recorded values were checked to ensure they matched the expected categories and response scales. Demographic variables were compared against the dataset documentation, while survey items were verified to contain only valid Likert scale responses (1–5). This confirms that no invalid or improperly encoded values are present."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "id": "WK_bcRn6-MZK",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 958
        },
        "id": "WK_bcRn6-MZK",
        "outputId": "74c77a23-079d-413d-ff0f-5a0e0403d520"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Variable</th>\n",
              "      <th>Unique Count</th>\n",
              "      <th>Unique Values</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Gender</td>\n",
              "      <td>3</td>\n",
              "      <td>[1, 2, 3]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Income</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Area</td>\n",
              "      <td>3</td>\n",
              "      <td>[1, 2, 3]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Frequently</td>\n",
              "      <td>4</td>\n",
              "      <td>[1, 2, 3, 4]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>PU1</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>PU2</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>PU3</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>PU4</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>PEU1</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>PEU2</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>PEU3</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>11</th>\n",
              "      <td>FSC1</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12</th>\n",
              "      <td>FSC2</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>13</th>\n",
              "      <td>FSC3</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>14</th>\n",
              "      <td>SP1</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>15</th>\n",
              "      <td>SP2</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>16</th>\n",
              "      <td>SP3</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>17</th>\n",
              "      <td>SP4</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>18</th>\n",
              "      <td>TP1</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19</th>\n",
              "      <td>TP2</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>20</th>\n",
              "      <td>TP3</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>21</th>\n",
              "      <td>IB1</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>22</th>\n",
              "      <td>IB2</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>23</th>\n",
              "      <td>IB3</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>24</th>\n",
              "      <td>IB4</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25</th>\n",
              "      <td>AUB1</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>26</th>\n",
              "      <td>AUB2</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>27</th>\n",
              "      <td>AUB3</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>28</th>\n",
              "      <td>AUB4</td>\n",
              "      <td>5</td>\n",
              "      <td>[1, 2, 3, 4, 5]</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "      Variable  Unique Count    Unique Values\n",
              "0       Gender             3        [1, 2, 3]\n",
              "1       Income             5  [1, 2, 3, 4, 5]\n",
              "2         Area             3        [1, 2, 3]\n",
              "3   Frequently             4     [1, 2, 3, 4]\n",
              "4          PU1             5  [1, 2, 3, 4, 5]\n",
              "5          PU2             5  [1, 2, 3, 4, 5]\n",
              "6          PU3             5  [1, 2, 3, 4, 5]\n",
              "7          PU4             5  [1, 2, 3, 4, 5]\n",
              "8         PEU1             5  [1, 2, 3, 4, 5]\n",
              "9         PEU2             5  [1, 2, 3, 4, 5]\n",
              "10        PEU3             5  [1, 2, 3, 4, 5]\n",
              "11        FSC1             5  [1, 2, 3, 4, 5]\n",
              "12        FSC2             5  [1, 2, 3, 4, 5]\n",
              "13        FSC3             5  [1, 2, 3, 4, 5]\n",
              "14         SP1             5  [1, 2, 3, 4, 5]\n",
              "15         SP2             5  [1, 2, 3, 4, 5]\n",
              "16         SP3             5  [1, 2, 3, 4, 5]\n",
              "17         SP4             5  [1, 2, 3, 4, 5]\n",
              "18         TP1             5  [1, 2, 3, 4, 5]\n",
              "19         TP2             5  [1, 2, 3, 4, 5]\n",
              "20         TP3             5  [1, 2, 3, 4, 5]\n",
              "21         IB1             5  [1, 2, 3, 4, 5]\n",
              "22         IB2             5  [1, 2, 3, 4, 5]\n",
              "23         IB3             5  [1, 2, 3, 4, 5]\n",
              "24         IB4             5  [1, 2, 3, 4, 5]\n",
              "25        AUB1             5  [1, 2, 3, 4, 5]\n",
              "26        AUB2             5  [1, 2, 3, 4, 5]\n",
              "27        AUB3             5  [1, 2, 3, 4, 5]\n",
              "28        AUB4             5  [1, 2, 3, 4, 5]"
            ]
          },
          "execution_count": 24,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "unique_summary = validate_unique_values(scommerce_df)\n",
        "\n",
        "unique_summary"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "JYIuxau3AJRA",
      "metadata": {
        "id": "JYIuxau3AJRA"
      },
      "source": [
        "The results confirm that all variables are within their expected format and range. No invalid values were identified; therefore, no observations were removed."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "pthZNl66BVM8",
      "metadata": {
        "id": "pthZNl66BVM8"
      },
      "source": [
        "### Composite Score Creation\n",
        "\n",
        "Composite scores were created by averaging the survey items corresponding to each construct, producing a single representative score for analysis. This approach is consistent with the methodology used by the dataset authors."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "id": "vlY5r8heEVjB",
      "metadata": {
        "id": "vlY5r8heEVjB"
      },
      "outputs": [],
      "source": [
        "# Perceived Usefulness\n",
        "scommerce_df = compute_composite_score(\n",
        "    scommerce_df,\n",
        "    ['PU1', 'PU2', 'PU3', 'PU4'],\n",
        "    'PU'\n",
        ")\n",
        "\n",
        "# Perceived Ease of Use\n",
        "scommerce_df = compute_composite_score(\n",
        "    scommerce_df,\n",
        "    ['PEU1', 'PEU2', 'PEU3'],\n",
        "    'PEU'\n",
        ")\n",
        "\n",
        "# Familiarity with Social Commerce\n",
        "scommerce_df = compute_composite_score(\n",
        "    scommerce_df,\n",
        "    ['FSC1', 'FSC2', 'FSC3'],\n",
        "    'FSC'\n",
        ")\n",
        "\n",
        "# Social Presence\n",
        "scommerce_df = compute_composite_score(\n",
        "    scommerce_df,\n",
        "    ['SP1', 'SP2', 'SP3', 'SP4'],\n",
        "    'SP'\n",
        ")\n",
        "\n",
        "# Trust in Platform\n",
        "scommerce_df = compute_composite_score(\n",
        "    scommerce_df,\n",
        "    ['TP1', 'TP2', 'TP3'],\n",
        "    'TP'\n",
        ")\n",
        "\n",
        "# Interaction Behavior\n",
        "scommerce_df = compute_composite_score(\n",
        "    scommerce_df,\n",
        "    ['IB1', 'IB2', 'IB3', 'IB4'],\n",
        "    'IB'\n",
        ")\n",
        "\n",
        "# Actual Usage Behavior\n",
        "scommerce_df = compute_composite_score(\n",
        "    scommerce_df,\n",
        "    ['AUB1', 'AUB2', 'AUB3', 'AUB4'],\n",
        "    'AUB'\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "k7cvN__9HRut",
      "metadata": {
        "id": "k7cvN__9HRut"
      },
      "source": [
        "The PU questionnaire items are averaged row-wise (`axis=1`) to create one composite score per respondent, which is stored in a new `PU` column. The same procedure is applied to the remaining constructs (PEU, FSC, SP, TP, IB, and AUB)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 26,
      "id": "T6js7Yy5HN6N",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "T6js7Yy5HN6N",
        "outputId": "06386e09-7a99-4157-f08e-c1c2a31907f1"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PU</th>\n",
              "      <th>PEU</th>\n",
              "      <th>FSC</th>\n",
              "      <th>SP</th>\n",
              "      <th>TP</th>\n",
              "      <th>IB</th>\n",
              "      <th>AUB</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>4.00</td>\n",
              "      <td>4.0</td>\n",
              "      <td>3.00</td>\n",
              "      <td>4.00</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.00</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.00</td>\n",
              "      <td>3.00</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.00</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.00</td>\n",
              "      <td>3.25</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5.0</td>\n",
              "      <td>5.0</td>\n",
              "      <td>5.0</td>\n",
              "      <td>5.00</td>\n",
              "      <td>5.0</td>\n",
              "      <td>5.00</td>\n",
              "      <td>5.00</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>2.75</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.75</td>\n",
              "      <td>3.00</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "    PU  PEU  FSC    SP   TP    IB   AUB\n",
              "0  4.0  4.0  4.0  4.00  4.0  3.00  4.00\n",
              "1  3.0  3.0  3.0  3.00  3.0  3.00  3.00\n",
              "2  3.0  3.0  3.0  3.00  3.0  3.00  3.25\n",
              "3  5.0  5.0  5.0  5.00  5.0  5.00  5.00\n",
              "4  4.0  4.0  3.0  2.75  3.0  3.75  3.00"
            ]
          },
          "execution_count": 26,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df[['PU', 'PEU', 'FSC', 'SP', 'TP', 'IB', 'AUB']].head()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "32b1gMHhg8N_",
      "metadata": {
        "id": "32b1gMHhg8N_"
      },
      "source": [
        "The first five rows of the composite score variables are displayed to verify the computed scores."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "RMQuiOvW4e-V",
      "metadata": {
        "id": "RMQuiOvW4e-V"
      },
      "source": [
        "Following the methodology of the original research paper, questionnaire items were averaged to create composite scores for each construct. These representative variables (`PU`, `PEU`, `FSC`, `SP`, `TP`, `IB`, and `AUB`) simplify subsequent analyses and are used throughout the remainder of the notebook."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "5b336f80",
      "metadata": {
        "id": "5b336f80"
      },
      "source": [
        "### [3] Exploratory Data Analysis"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "lxJSly9_-Vf-",
      "metadata": {
        "id": "lxJSly9_-Vf-"
      },
      "source": [
        "#### **1. What does the demographic composition of the sample look like, and is it skewed in any way?**\n",
        "\n",
        "The demographic variables (`Gender`, `Income`, `Area`, and `Frequently`) are analyzed to describe the sample and identify any overrepresented groups that may influence the interpretation of subsequent analyses."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 27,
      "id": "hbZl966ByoBD",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "hbZl966ByoBD",
        "outputId": "68b87b03-b4d4-46db-a86d-d3bedd4aae27"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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HPOCMaprLAqk67r2dUrwAjelSpWKs/zQoUMuqKiyHHv37nX7RsFlBeYUjFNg62zeV8G++eYbV7NVtZPr1KljF1xwgSv3oZIO8SfXU8BMgVEF6FTLV49frFgxFwxs3bq1CzImZsKECda7d28XhNd+1jZ9+OGH7v22Y8eOOLdVjddHHnnE1fpV4FEBRgV8FYRNzJnuo+B9paCZ7nPNNde440cT/t17772BALTaN7F2966fK5VM0Wv02sGj/aTgo4LVOi50GwVO1X4KKuqSGC8IrSCzAqfaTgWuNfHWK6+84h5T72m9Xr3/FbBUuwRTezRp0sR9Hhw9etS1sT6H9Jmm/aXjJZgeT8FIfZ4o6KpjWLfX/lX7qn0So4Cu2uf88893Ac9s2bK595KW6706fvx4F7xVe+q16Hk++OADd1JFx2h8ake9BzW5mAK22gbtU32+JHXiSkFrXXLnzu32iTLpFaDV4ygD+nS0j/R+UYa8tt3br9pHOqa8zy4FbPVZ8+6777rjS/u0ePHirr20T8/0ZBMAAEhDsQAAwLe++eab2NKlS8def/31p73tokWL3G1bt24dWPbCCy+4ZaNHjw4su/HGG2PLly8fu3HjxsCykydPxg4dOtTd9vHHHw8s37Fjh1t27bXXxnkub7kugwYNCiz/999/3V89hta9+eabgXXeMl1mzJgRWH7kyJHYDh06uOX9+vULLJ83b55b9thjjyX6er3H+vvvvxM8R/DzJrV8+PDhblmLFi1i9+3bF1geHR0d26xZM7dOt4nfFro88sgjsYcPHw6se+2119zyq666yu3L0+nWrZu7fefOnWMPHToUWP7rr7+69tE6PWYwvQe0XLdJjnvuucfdXvsxvuPHj8du27YtdvLkybEVKlRwt1uyZEmi+6xHjx5xXuvWrVsD2/LVV1+d1fvKa1tdtC+OHTsWuP2oUaPc8ttvvz3wftJ6vQe1fMKECXH28WeffRZbqVIl99zr169PsP3lypWLfe+99wLL9VitWrVy60aMGBFYvmbNmtgyZcrE1qxZM/bnn38OLN++fXts3bp1A9t7LvvIOx51mTZtWuB16HV2797dLX/ooYfOqd295wje30kZOHCgu+2YMWMS3L9t27Zxjou9e/fGNmnSJMFxFNyW2p/eazp48GBsnTp1AuveeuutwH1++eUX1y7a395z6H533HFH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7YsvvrCJEydapkyZQr2pAAAA8KHzzjvPVq9e7QKoF198cZx1R44csU8//TTVnnvZsmWu3woASBsEbQEgjTJsFbC99tprbezYsZYly/8+fuvUqWO1atWy7t272/vvv28NGzakTQAAAJBA+fLlbdOmTbZ48WK777774qxTwDZHjhyWN29e9hwAZACURwCANDB16lSXGTFo0KA4AVuPyiU0adIkcP3kyZM2ZcoUu+mmm6xixYpu/euvvx7nPhpm1rdvX3e7unXrWqVKldxwubVr18a53fLly61ly5auHIMeR1kS8R0/ftxGjhzpAsh6vttuu81lBgfTkDRlBN97771WuXJl99wAAABIOzlz5nT9NQVt41PfTX29+H1N9fMmTJhg9evXd/3Fm2++2fUf1d9Mbr9y/vz5LgFBbrzxxjglEf7++2/Xj7zmmmusSpUq1q5dO9u2bVui268kheuuuy7Oc4ueW9ueXCrzpX7yZ5995vqtXn/5rbfeinO7PXv22OOPP25XXXWVVa1a1e655x5btWrVGe+b/v37uxFxSsBQn/qBBx6w6OhomzdvntsOPbaC6L/99luc5//oo4+sadOm7rG1f4YOHeoyogEgOQjaAkAa+Pjjj+3KK6+0iIiIJG/zzDPPBLJsBw4caC+88ILdfvvt9uKLL7qOpAKm6lQG++CDD9xjP/XUUzZ69GjXeXzooYfs33//devXr1/vOs558uRxj9e2bVvr0aNHgskkunbtarNmzXJlGyZNmuQ6nqqJFr/j+8Ybb7hOpzqtzZs3T8E9BAAAgORQf9ErkeA5dOiQff7559aoUaME/bzOnTu7BIIWLVoE+pUa+TVgwIBk9ysVyH3wwQfd7caPH29dunSJEyzeuHGjjRgxwj3munXrXD8yMeo/7t6927799tvAMpULUxD6jjvuOKM3wN69e23w4MGuf6tAa5EiRVyAViPc5PDhw9a6dWv3XL169XLbnS1bNtc3/vXXX89o37z77rv29ddf27Bhw1yAWf8rAPzaa6+559R2rFmzxv31vPPOO66Pffnll7s+fLdu3WzhwoVu3zGZG4DkoDwCAKSyv/76y12KFy+eYF38ycdUz3b79u325ptvuuCqJnuQ2rVru3WTJ0+2u+66yy666KLA/V9++WXLnTt3oHOqjuOPP/7oMg50ewWKFYg9//zz3W103+COtDJvVU93zJgxgaCxsgiOHj1qzz33nOv8exkbhQsXtp49e6bavgIAAMCpKYCqMgjBJRKWLFni+nya5DaYArnq6ykIe+utt7plyvjMnj27Pf/88y7gWapUqWT1Ky+99FK3vFy5ci5A6ilYsKA7oe/1NZVlq76nAsneY3nUp1UtXiUGKPvV23ZlnwaPOksO9VUVRPUeR33t66+/3tXd1RwSCxYscJOm6a+2WapVq+ae57vvvnPbeSb7RkHfCy64wF3/8MMPXf9ZmbRFixZ1yxRIf/vtt93/CsqqH60+tf56tI1qM22j2hEAToVMWwBIZfGHf3nUUaxQoUKci4ZXffPNN66jp3IE6iB6F13XEK4VK1YEHqNkyZJxOsPqNHudWNFt1Vn0OtGiYV+ZM2cOXFemgALCGmoX//mUwaDMCY/X4QUAAEBoKKioflpwiYT33nvPGjRokGBCW5XJ0sl3ZZAG02gub31y+5VJUdms4L6mF9A9cOBAgtuqXJgyahX09B5XQdWrr746wcRqyaFyDB7v/l75AfWDtS3B/VcFu5VRrMzaM9k3CgJ7AVuJjIx0iRBewFYuvPBCO3jwoPt/y5YtLhM6fn/+iiuucPv4q6++OuPXCiD8kGkLAKlMHTrVH9OZ/mCFChWyuXPnBq5r2NQvv/xi+/fvd9e9M/7xaUhZcMczfkc4OFCsDF8vK9ejzmnwMj2fgsTKPEiMaoF5nV29DgAAAISWArQabq/AoIb86yT8I488kuB2Xl8w+IS95M+f3/31gozJ6VcmJX7/8HT3a9asmStFoMCtyodp24OzUc9E8DZ7z+uVHlAf91Slyc5k38TPGD5dv9jrz2s+C10S618DwOkQtAWANKCz7JrRN3iYWNasWV192OCz8+LN+Pvqq69arly5EjyWShQklx5T9ciCqSOrTqpH9W7V6VRNrsQUK1Ys2c8HAACA1KfJvNRPVLat+nHKKFUJg/iUHfrnn3+6urTBwUkvaBj/5H5aUHZqzZo17f3333fBTfWN69Wrl+LPoz5u/InBZOXKlW6/pOa+8frzvXv3dq81vuCsXQBICuURACANqDathkRpYocTJ04kWK8JGHbs2OH+r1GjhvurTqSCut4lJibG1dfyztwnh2p8qZZZ8LA21d/SLL8edSQ1jEzB3ODnU9avsn/j190FAABAaOnkvwKdGuqv4GdSI7TUz1NfLriUgmhCLIlfA/dUvEzWlKAJyVRPVhN8aU4FZQunNPWp1b8OLvWlUmOaXE2j3VJy38SnyceU5augcXD/WiUnRo0aZRs2bDiHVwYgXJBpCwBpoEyZMvbss89anz59rGnTpq6jqmXqKK5atcp1HJUR26FDB7dctbT69evnSiooa2Lr1q1uojBlUSQ2oVlSNGOtJkho3769e2wFfjUjbnDdMdWyVX0tzWSri2p2rV271l544QVXDzdfvnyptFcAAABwthTs7NSpkwumKjEgqYzcWrVqufUqsVW2bFlXq/Wll15ytWVVx/ZMs0c1cZgeV33Gs3XLLbfYkCFDXJ9Tfd7UoD7366+/bg8++KB1797dZc5qZJmSFzSxr/rVKbVv4lPmrib+7d+/v/tfE6Spxq8mbNNzaS4LADgdgrYAkEbUOVUAdubMmS5Iq4Cssls1REyd7latWgUCssOHD7fJkyfbrFmzXK0ynanXbVSrLH7drVPR482YMcNGjBjhOo56HM0CrOsedfSnTJnisnj1nPv27XNZAPfff78L+gIAAMB/NHmXAqmaJyGpAKomJlP/TifjX3nlFXcCX8HKHj16uL7emVCAU8+pTFHVoVX/8Wwps1b1bDVhlyYySw0qu6B+8MiRI12AWDV2NXGZArfeBGIptW8So8nOVMJi6tSpNnv2bFfGQnNIqH5v8ARmAJCUTLFelW4AAAAAAIBUptJgGu2lUV733nsv+xsAEkGmLQAAAAAASHUaabZgwQJXz1ZZwM2aNWOvA0ASCNoCAAAAAIBUp7JcqjOrsgGar0ElDAAAiaM8AgAAAAAAAAD4yHmh3gAAAAAAAAAAwP8QtAUAAAAAAAAAHyFoCwAAAJyl+fPnW5kyZRJcypYt69Zv2LDBWrRoYVFRUW7CnXXr1sW5/7vvvmv16tVz67t27WoxMTG0BQAAAKhpCwAAAJytY8eO2cGDBwPX//nnH7v33nutbt269sgjj9jNN99st912mzVv3txmzpxp77//vi1ZssRy5sxpa9eutTZt2tigQYNckHfYsGFu+eTJk2kQAACAMEemLQAAAHCWsmfPbvnz5w9cFi5caLGxsdazZ09btGiRZcuWzXr37m0lSpSwvn37uhnTFy9e7O47Y8YMa9CggTVp0sQFbUeOHGlLly61HTt20B4AAABhLouFib17/5cBAX/Kly+XxcQcDvVmAOkaxxHAcRQO8ufPY360f/9+e+mll2zo0KGWNWtWW7NmjVWvXt0yZcrk1utvtWrVbPXq1da0aVO3/oEHHgjcv1ChQla4cGG3vGjRogkeP5z7s3y/hSfaPTzR7uGJdg9P4dzu+ZPRnw2boC38Tb9lMmc+z/2NjQ311gDpE8cRwHGE0FL5gwIFClj9+vXd9b1791rJkiXj3CYiIsI2btzo/t+zZ4+7ffz1u3btSvI5/j/+G5bfb+edRz8xnNDu4Yl2D0+0e3ii3U+PoC0ApANLl35qffv2irOsbt0bbOhQDaX91KZMmWB79uy2cuXKWbduPax06f9OgHP8+HGbOPF5+/jjJe76ddfVtYce6mE5cuQIyesAgIxKJRHmzJljHTp0CCw7evSoy7gNpusnTpwI1MM91frEslEUvMxI9FqHDx/uJmQ7//zzXe3fRx991Nq2bWvLly9PcHtlKOv2ut+YMWPsvffec/u5Zs2a1q9fP7v44otD8jqQeiIi/JlZj9RFu4cn2j080e5JI2gLAOnAr79usWuuudZ69+4bWJY1azbbsmWzDRr0lPXq1ccqV65iCxfOsV69HrbZs992dRanT3/JVq9eac8997wLKAwbNtAmT55gjzzSM6SvBwAymh9++MF2795tt956a2CZ6tnGD8Dquj6fT7U+qRNrGj6Y0TJtR4582lau/N5GjXrBjhw5YgMGPGl58+azQYOG299//+1uo9e8fftmN7Fbw4ZNLDr6oE2aNM4+/fRj69dvsF144UU2ceIL1rnzg/bSS68GylEgfVMz6of8vn0HGYkXRmj38ES7h6dwb/fISJ+XR0jqzLo6Whs2bLABAwbYL7/84oaVaVbdihUrBu6r+4wdO9YNO6tdu7YNGTLE8uXLF8qXAwCpZtu2X+3yy0taRERknOWLFi20yy673Bo0aOS+9Hr06GFvvPGGC/KWLVvevv76K7v99jvc/9KkSTN7++35tBQApLAvvvjCatSoYRdccEFgWcGCBS06OjrO7XTdK4mQ1HpNaJaUjPSj5sCBv+zdd9+2sWMnWrly/+3nt2x5j61fv84aN24WuN3Jk/+6rNq77mrrvs+0DxYtetcefvgxq1KlurtN795PWZMm9d0kbkWLXhqy14SUp/bOSO97JA/tHp5o9/BEuyctpOOrNEnDsmXL7OWXX7ZRo0bZm2++abNnz3Zn2Tt27Og6vvPnz7eqVatap06d3HJZu3atm323W7du7vYHDhywPn36hPKlAECqUhA2sR+hefNeYFu3brG1a1fbyZMn3WemZiYvXLiIW6/ggTKR9Dmpi0oplC5dhtYCgBSm/qkmGQsWFRVlq1atciMdRH9XrlzplnvrV6xYEbj9H3/84S7e+oxO3125c+e2qlX/G3iVNm3usyefHBDndosWvWN//fWX3XPPve66vu+UYXvFFbUSPObhw4fSYMsBAABSX5ZQzq47b948mz59ulWuXNkta9eunZstN0uWLG64WO/evV3WrQK0n3/+uS1evNjVsZoxY4Y1aNDAmjRp4u43cuRIu/766///zHrCmXYBID3Tj/zt27fZt99+ba+9Nt1lHF1/fT3r0KGz3XjjzfbVV59bly4dLHPmzHbeeefZyJFjLG/evO6+Xbo87Grh3nrrje66snWfeWZ0iF8RAGQ8mlzs9ttvj7NME5IpMWHYsGHWqlUrmzVrlqu/qn6stG7d2tq0aWNVqlSxSpUqudvVrVs3bPqzO3f+bhdfXNjef/9de/316fb33//YrbfeZm3btnPfZ9534BtvvOZq3ObMmdNl42hd/IDtnDkz7cILL7QSJUqF6NUAAABkkExbZRXozLomDfAou1blEhS4rV69eqAelf4qc2H16tXuutYrC9dTqFAhK1y4sFsOABnN7t27ApPVDBky3Lp2fcSWLFlsEyY874aW7tu3zx59tLdNmfKKNW7c2J5+erD9+WeMu+/vv++wggUvtuefn2SjR4+zEyeO27hxY0L9kgAgw1FZA++EmUd93cmTJ7t+rxIP1FedMmWKCz6KRpMNHjzYJkyY4AK4Gh2hvnC40Ci6337bbgsXznfZtd26PWxz586y2bP/E7jNqlUr3ESbd955Z5KP88UXn9msWTOsU6duruQaAABARhCyTFtlxV5yySX21ltv2YsvvugmGlBn9sEHH3R1alXHNlhERITLYJA9e/YEaoEFr9+1a1eavgYASAsXX1zIFi362PLkyetOYpUqVcZiY0/a4MH97eDBv6xEiZLWrNmdrqbt1VfXsJtvvsXee2+h3XFHcxsxYoiNHTvJKlT4b63APn36W7duHa19+84WGRm3Pi4A4NzKIyRGI8oWLFiQ5P3U/9UlHGXOnMUOHz5sAwYMc9913onK+fPnWuvW97jr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UqVKlutly5Z1P6tXr+4WFXvqqads9OjRdsMNN9js2bNdndsOHTq42/To0cN69uxpDRs2tAYNGrjbtWrVyqpVq3baxwUAAIBiGrRNTk4+4f4xY8ac6vEAAAAABa6g+7PlypWzKVOmuIXKXnvtNatdu7ZNnTrVypQp4/arpMKoUaPsueees//85z92+eWX26OPPnpafxMAAADFPGibnaaFbd++3b788ku7+eab8+MhAQAAgEKTH/3ZJ554Isv1iy66yObPn5/r7VUawSuPAAAAAJx20Da3zIMXXnjBvv7661N5SAAAAKDQ0J8FAACAn52Rnw/Wvn17e/fdd/PzIQEAAIBCQ38WAAAAYRW0PXz4sKvXdc455+TXQwIAAACFhv4sAAAAinR5hAsvvNAiIiKO2x4VFWWPPfZYfhwXAAAAUGDozwIAACDsgrbTp0/Pcl0B3DPPPNNq1arlVsoFAADwk9TUPfbnP4+3lSs/d4PMbdpcaX37DnS/r1v3b5s48RnbsmWTxcbGWb9+fa116/aB+y5atMBmznzZPUaNGufboEH32EUXNQzp/4PTR38WAAAAYRe0bdasmfv57bff2pYtW+zXX3+18847j4AtAADwnczMTHvooaFWvnx5mzRpmh04sN/GjBllZ5wRaTfccJMNHnynXXNNdxs2bIR9/fWX9uijoywqqpxdemlz+/jj5fb002Nt6NCHrG7d+vb3vy+0+++/y2bOnOcCvCi66M8CAAAg7IK2+/fvt+TkZHvvvffsrLPOsl9++cUOHTpkF198sU2aNMmdFAEAAPjBd99ts/Xr/20LFrxj0dExbtutt/azSZP+bFWqVLGYmBjr12+g237uuefahg1rbcmSt13Q9u9/f8s6dOhkV13Vwe2//fY77P3337Xly5fZn/50TUj/L5we+rMAAAAIu4XIVLd29+7dtnjxYvvkk0/s888/t7feesst3jBmzJj8P0oAAIBTpEDtU09NCARsPYcOHbTExMssOfmR4+6jfXLjjf9n119/03H7Dx78734UXfRnAQAAEHZB2/fff99GjBhh559/fmCb6tkOHz7cZd8CAAD4hWYAJSZeGriusk6vv/6aNWlysVWqVNnq128Q2Pfjj+m2aNEit09q177QqlU7N7Bf5RK2b/8usB9FF/1ZAAAAhF15BC3accYZx8d7tSCZSiUAAAD41eTJ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",
            "text/plain": [
              "<Figure size 1400x1000 with 4 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "\n",
        "# Count observations\n",
        "gender_counts = scommerce_df[\"Gender\"].value_counts().sort_index()\n",
        "income_counts = scommerce_df[\"Income\"].value_counts().sort_index()\n",
        "area_counts = scommerce_df[\"Area\"].value_counts().sort_index()\n",
        "freq_counts = scommerce_df[\"Frequently\"].value_counts().sort_index()\n",
        "\n",
        "# Replace numeric codes\n",
        "gender_counts.index = gender_counts.index.map({\n",
        "    1: \"Male\",\n",
        "    2: \"Female\",\n",
        "    3: \"Different\"\n",
        "})\n",
        "\n",
        "income_counts.index = income_counts.index.map({\n",
        "    1: \"< $100\",\n",
        "    2: \"$100-$200\",\n",
        "    3: \"$200-$300\",\n",
        "    4: \"$300-$400\",\n",
        "    5: \"> $400\"\n",
        "})\n",
        "\n",
        "area_counts.index = area_counts.index.map({\n",
        "    1: \"Urban\",\n",
        "    2: \"Suburban\",\n",
        "    3: \"Rural\"\n",
        "})\n",
        "\n",
        "freq_counts.index = freq_counts.index.map({\n",
        "    1: \"Daily\",\n",
        "    2: \"Weekly\",\n",
        "    3: \"Monthly\",\n",
        "    4: \"Rarely Used\"\n",
        "})\n",
        "\n",
        "# Create 2x2 subplot layout\n",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
        "\n",
        "# Gender\n",
        "ax = axes[0, 0]\n",
        "gender_counts.plot(kind=\"bar\", ax=ax)\n",
        "\n",
        "for container in ax.containers:\n",
        "    ax.bar_label(container)\n",
        "\n",
        "ax.set_title(\"Gender\")\n",
        "ax.set_xlabel(\"Gender\")\n",
        "ax.set_ylabel(\"Count\")\n",
        "ax.tick_params(axis=\"x\", rotation=0)\n",
        "\n",
        "# Income\n",
        "ax = axes[0, 1]\n",
        "income_counts.plot(kind=\"bar\", ax=ax)\n",
        "\n",
        "for container in ax.containers:\n",
        "    ax.bar_label(container)\n",
        "\n",
        "ax.set_title(\"Monthly Income\")\n",
        "ax.set_xlabel(\"Income\")\n",
        "ax.set_ylabel(\"Count\")\n",
        "ax.tick_params(axis=\"x\", rotation=20)\n",
        "\n",
        "# Residential Area\n",
        "ax = axes[1, 0]\n",
        "area_counts.plot(kind=\"bar\", ax=ax)\n",
        "\n",
        "for container in ax.containers:\n",
        "    ax.bar_label(container)\n",
        "\n",
        "ax.set_title(\"Residential Area\")\n",
        "ax.set_xlabel(\"Area\")\n",
        "ax.set_ylabel(\"Count\")\n",
        "ax.tick_params(axis=\"x\", rotation=0)\n",
        "\n",
        "# Social Media Usage Frequency\n",
        "ax = axes[1, 1]\n",
        "freq_counts.plot(kind=\"bar\", ax=ax)\n",
        "\n",
        "for container in ax.containers:\n",
        "    ax.bar_label(container)\n",
        "\n",
        "ax.set_title(\"Social Media Usage Frequency\")\n",
        "ax.set_xlabel(\"Frequency\")\n",
        "ax.set_ylabel(\"Count\")\n",
        "ax.tick_params(axis=\"x\", rotation=20)\n",
        "\n",
        "# Overall formatting\n",
        "fig.suptitle(\"Distribution of Respondent Demographics\", fontsize=16)\n",
        "\n",
        "plt.tight_layout(rect=[0, 0, 1, 0.96])\n",
        "\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "J2dM9e3-sbi2",
      "metadata": {
        "id": "J2dM9e3-sbi2"
      },
      "source": [
        "The sample is predominantly **female**, has **income below $100**, resides mainly in **urban areas**, and reports **daily social media usage**. Overall, the respondents represent digitally active university students, providing an appropriate sample for examining social commerce behavior."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "fpKndzFE39_0",
      "metadata": {
        "id": "fpKndzFE39_0"
      },
      "source": [
        "#### Summary\n",
        "\n",
        "The sample primarily consists of female university students with low income, most of whom live in urban areas and use social media daily. These characteristics are consistent with the study's target population and provide appropriate context for interpreting the subsequent analyses of social commerce behavior."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "DH5EbxQi-Ade",
      "metadata": {
        "id": "DH5EbxQi-Ade"
      },
      "source": [
        "#### **2. How are each of the constructs (PU, PEU, FSC, SP, TP, IB, AUB) distributed among respondents?**\n",
        "\n",
        "Descriptive statistics and histograms are used to examine the distribution of the composite scores, allowing the central tendency, variability, and overall response patterns of each construct to be assessed."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "id": "M5R5sFNvkODv",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "M5R5sFNvkODv",
        "outputId": "987b1a3b-d0c2-4aad-eb5a-ab2441bcd76c"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PU</th>\n",
              "      <th>PEU</th>\n",
              "      <th>FSC</th>\n",
              "      <th>SP</th>\n",
              "      <th>TP</th>\n",
              "      <th>IB</th>\n",
              "      <th>AUB</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>757.000000</td>\n",
              "      <td>757.000000</td>\n",
              "      <td>757.000000</td>\n",
              "      <td>757.000000</td>\n",
              "      <td>757.000000</td>\n",
              "      <td>757.000000</td>\n",
              "      <td>757.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>3.773448</td>\n",
              "      <td>3.700132</td>\n",
              "      <td>3.730956</td>\n",
              "      <td>3.601717</td>\n",
              "      <td>3.557023</td>\n",
              "      <td>3.606011</td>\n",
              "      <td>3.676684</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>0.702081</td>\n",
              "      <td>0.710670</td>\n",
              "      <td>0.749862</td>\n",
              "      <td>0.701395</td>\n",
              "      <td>0.730930</td>\n",
              "      <td>0.703760</td>\n",
              "      <td>0.707042</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>3.250000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>3.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>4.000000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>3.750000</td>\n",
              "      <td>3.666667</td>\n",
              "      <td>3.750000</td>\n",
              "      <td>3.750000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>4.000000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>4.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>5.000000</td>\n",
              "      <td>5.000000</td>\n",
              "      <td>5.000000</td>\n",
              "      <td>5.000000</td>\n",
              "      <td>5.000000</td>\n",
              "      <td>5.000000</td>\n",
              "      <td>5.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "               PU         PEU         FSC          SP          TP          IB  \\\n",
              "count  757.000000  757.000000  757.000000  757.000000  757.000000  757.000000   \n",
              "mean     3.773448    3.700132    3.730956    3.601717    3.557023    3.606011   \n",
              "std      0.702081    0.710670    0.749862    0.701395    0.730930    0.703760   \n",
              "min      1.000000    1.000000    1.000000    1.000000    1.000000    1.000000   \n",
              "25%      3.250000    3.000000    3.000000    3.000000    3.000000    3.000000   \n",
              "50%      4.000000    4.000000    4.000000    3.750000    3.666667    3.750000   \n",
              "75%      4.000000    4.000000    4.000000    4.000000    4.000000    4.000000   \n",
              "max      5.000000    5.000000    5.000000    5.000000    5.000000    5.000000   \n",
              "\n",
              "              AUB  \n",
              "count  757.000000  \n",
              "mean     3.676684  \n",
              "std      0.707042  \n",
              "min      1.000000  \n",
              "25%      3.000000  \n",
              "50%      3.750000  \n",
              "75%      4.000000  \n",
              "max      5.000000  "
            ]
          },
          "execution_count": 28,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Summary statistics for the composite constructs\n",
        "constructs = ['PU', 'PEU', 'FSC', 'SP', 'TP', 'IB', 'AUB']\n",
        "\n",
        "# Generate descriptive statistics\n",
        "scommerce_df[constructs].describe()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "82vMIj-UqfNs",
      "metadata": {
        "id": "82vMIj-UqfNs"
      },
      "source": [
        "Across all constructs, mean scores ranged from **3.56** to **3.77**, indicating generally positive responses."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 29,
      "id": "LU-SW6JblNfT",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "LU-SW6JblNfT",
        "outputId": "536acd72-23ed-4443-a5df-33c599b9fd19"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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vWbMmGGWUd57773hHl7ECQQCJlyxlprOmzfPBFR333236ZN26aWXhnt4Eeu9994zO2teccUV4R5KRCsqKtKECRM0a9as/TaGAAAAAIBI06Mrpbr6D7trrrlGO3bsCPdQIt5XX31lentZ1VFDhw7V6aefbkKqd999N9xDi0hVVVWmJ5q19HTAgAE699xzNW7cONMTDYdWV1en2bNnmyAPR7Z161YNGzZMOTk57QebPwAAAACIRIRSh7F06VKdddZZev3117v3E+mBrD96reVn2dnZ+73e2NgYtjFFstzcXD3xxBNKTk42VWZWGLVs2TKz9BGHZlVDfvvb39aQIUOYomMIpaywEwAAAAAiXY9evteVrrvuunAPocewqjCsPlL7Lk975ZVXNHbs2LCOqyc477zzVF5ebpZbXXjhheEeTkRavHixli9frnfeecdUl+HwrJBz27Zt+vzzz/XMM8/I7/dr8uTJpnIxNjaWqQMAAAAQUaiUQsg99thjWr9+ve666y5m9yh+//vfa86cOdqwYYNZ/oiD+7r993//t37+858rPj6e6TkKK+BsaWkxAZRVjWf1dLPCPGvpIwAAAABEGiqlEPJA6qWXXtLvfvc709cGR7a3R5IVvvz0pz/V9OnTqWjZx1NPPaVTTjllv0o8HF5+fr7ZKTQtLU02m830LrMqF++55x6zW6nD4WD6AAAAAEQMQimEjLXb19y5c00wxVK0Izc6txrDT5w4sf01q1eS1+s1fbgyMzP5qdxnxz1rvgoLC81zj8djzh9++KFWrVrFPB1Cenr6fs8HDx5sQk9rV0x+tgAAAABEEkIphKyi5bXXXtPjjz9uetjg8EpLS3Xbbbdp0aJFysvLM6+tW7fOBAaEBvt7+eWX5fP52p//5je/MWerqgwH++yzz8zcfPLJJ0pISDCvWUtDraCKny0AAAAAkYZQCiHZ7evpp5/WtGnTNGbMGFVWVu63Mx8OXrI3cuRIzZgxwyypKisrM9VlP/zhD5mqQyxH21dSUpI59+/fn7k6BKuiLC4uTjNnztStt96qkpIS00/qlltuYb4AAAAARBxCKRy3jz76yOzy9cc//tEc+9q0aRMzfACrr48V4lnLHadMmWIqWm644QZNnTqVucJxSU5O1gsvvKBHHnlEV111lQnxrr32WkIpAAAAABHJFrT2EAcAAAAAAAC6kb073wwAAAAAAACwEEoBAAAAAACg2xFKAQAAAAAAoNsRSgEAAAAAAKDbEUoBAAAAAACg2xFKAQAAAAAAoNsRSgEAAAAAAKDbEUoBAAAAAACg2xFKAV3gr3/9q4YPH64333xzv9fvu+8+cxyotLTUfL91ttxwww3m+d6jsLBQN998s4qLi/m8AAAAAAAnBEIpoAu899576tevn95+++1O3+Omm27S559/rs8++0xvvPGG0tPT9eMf/1jBYDCkYwUAAAAAIBwIpYAQq66u1uLFi3Xrrbdq+fLlKikp6dR9EhMTlZOTo9zcXA0dOtRUWBUVFWnTpk18ZgAAAACAHo9QCgixDz74QCkpKbrssstMoHQ81VL7SkhICMl9AAAAAACIBIRSQBcs3Rs/frzsdrvOO+88vfXWW8e95M7j8WjOnDntPaYAAAAAAOjpCKWAENq5c6dWrlypiRMnmueTJk0yy/dWrFjR4Xs988wzpsG5dYwePVrPP/+8brvtNtlsNj4zAAAAAECPFxPuAQAnWpVUXFycvvGNb5jnZ555ptLS0jRv3jydfvrpiomJMVVPB9pbSeV0Ottfu/baa80ufJbm5mYtWrRId999t5577jmNGzeu2/5NAAAAAAB0BUIpIMShVGtrq8aMGdP+mt/vN32mfvazn5leU9u3bz/oOpfLZc7W1/eywqz+/fu3Px8xYoSWLVumuXPnEkoBAAAAAHo8QikgRLZt26b169dr5syZOuuss9pft3bMu+uuu7RgwQLTD+rdd9+V1+vdrypqzZo1GjBggNlx70isiior5AIAAAAAoKejpxQQwiqp9PR0TZkyRcOGDWs/Lr74Yg0ZMsQ0PL/gggtMT6jp06dr48aNKi4uNq8/+eST+t73vrff/awle5WVleawelW9+uqrWrx4sS666CI+MwAAAABAj2cLHu+2YAAMKyw6++yzTaXUgV555RU9/PDD+uSTT+R2u/XYY4+ZpXhW8NSvXz9NnTpV11xzTfv3W72kli5d2v7cqqqylvJdf/31+u53v8uMAwAAAAB6PEIpAAAAAAAAdDuW7wEAAAAAAKDbEUoBAAAAAACg2xFKAQAAAAAAoNsRSgEAAAAAAKDbEUoBAAAAAACg2xFKAQAAAAAAoNsRSgEAAAAAAKDbEUoBAAAAAACg2xFKAQAAAAAAoNsRSgEAAAAAAKDbEUoBAAAAAABA3e3/AZ192YBgD90TAAAAAElFTkSuQmCC",
            "text/plain": [
              "<Figure size 1200x1000 with 7 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Plot distributions of the composite constructs\n",
        "fig, axes = plt.subplots(3, 3, figsize=(12, 10))\n",
        "axes = axes.flatten()\n",
        "\n",
        "for i, construct in enumerate(constructs):\n",
        "    sns.histplot(\n",
        "        scommerce_df[construct],\n",
        "        kde=True,\n",
        "        ax=axes[i]\n",
        "    )\n",
        "    axes[i].set_title(construct)\n",
        "\n",
        "# Remove unused subplots\n",
        "for j in range(len(constructs), len(axes)):\n",
        "    fig.delaxes(axes[j])\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "lsoJTVwbUR9I",
      "metadata": {
        "id": "lsoJTVwbUR9I"
      },
      "source": [
        "The histograms show similar distributions across all constructs, with responses concentrated around scores of 3 and 4. The larger peak near 4 indicates generally positive responses, while the smaller peak near 3 reflects more neutral evaluations. Overall, the distributions exhibit slight bimodal characteristics rather than perfect normality."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "tdaMoJ6WL3Pm",
      "metadata": {
        "id": "tdaMoJ6WL3Pm"
      },
      "source": [
        "#### **3. How are the behavioral/perception constructs (PU, PEU, FSC, SP, TP, IB) related to each other?**\n",
        "\n",
        "Pearson correlation analysis was performed on the composite construct scores (`PU`, `PEU`, `FSC`, `SP`, `TP`, `IB`, and `AUB`) to examine the relationships among the behavioral and perception constructs and their association with actual social commerce usage."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "zVoCZ6UJuIXn",
      "metadata": {
        "id": "zVoCZ6UJuIXn"
      },
      "source": [
        "Construct Score Computation\n",
        "\n",
        "The following code computes the average score for each construct:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "id": "8d9eQ7Gct8JD",
      "metadata": {
        "id": "8d9eQ7Gct8JD"
      },
      "outputs": [],
      "source": [
        "scommerce_df['PU'] = scommerce_df[['PU1','PU2','PU3','PU4']].mean(axis=1)\n",
        "scommerce_df['PEU'] = scommerce_df[['PEU1','PEU2','PEU3']].mean(axis=1)\n",
        "scommerce_df['FSC'] = scommerce_df[['FSC1','FSC2','FSC3']].mean(axis=1)\n",
        "scommerce_df['SP'] = scommerce_df[['SP1','SP2','SP3','SP4']].mean(axis=1)\n",
        "scommerce_df['TP'] = scommerce_df[['TP1','TP2','TP3']].mean(axis=1)\n",
        "scommerce_df['IB'] = scommerce_df[['IB1','IB2','IB3','IB4']].mean(axis=1)\n",
        "scommerce_df['AUB'] = scommerce_df[['AUB1','AUB2','AUB3','AUB4']].mean(axis=1)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "qprn6odlu-Z7",
      "metadata": {
        "id": "qprn6odlu-Z7"
      },
      "source": [
        "#### Correlation Matrix\n",
        "\n",
        "The table below presents the Pearson correlation coefficients between the composite constructs."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "id": "OV-8VM9NvALc",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OV-8VM9NvALc",
        "outputId": "42ea426d-c63c-4c8c-c1e3-c12bdec04420"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "           PU       PEU       FSC        SP        TP        IB       AUB\n",
            "PU   1.000000  0.788226  0.804070  0.718057  0.676910  0.655539  0.771895\n",
            "PEU  0.788226  1.000000  0.776458  0.717394  0.674808  0.691334  0.785858\n",
            "FSC  0.804070  0.776458  1.000000  0.766774  0.710337  0.657147  0.744687\n",
            "SP   0.718057  0.717394  0.766774  1.000000  0.734756  0.644206  0.691883\n",
            "TP   0.676910  0.674808  0.710337  0.734756  1.000000  0.663131  0.662269\n",
            "IB   0.655539  0.691334  0.657147  0.644206  0.663131  1.000000  0.683213\n",
            "AUB  0.771895  0.785858  0.744687  0.691883  0.662269  0.683213  1.000000\n"
          ]
        }
      ],
      "source": [
        "constructs = scommerce_df[['PU','PEU','FSC','SP','TP','IB', 'AUB']]\n",
        "\n",
        "corr_matrix = constructs.corr(method='pearson')\n",
        "\n",
        "print(corr_matrix)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "Haxpb04CvIYj",
      "metadata": {
        "id": "Haxpb04CvIYj"
      },
      "source": [
        "Correlation Heatmap\n",
        "\n",
        "To visualize these relationships more clearly, a heatmap was generated."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 32,
      "id": "PYbIAzvcvLMJ",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 538
        },
        "id": "PYbIAzvcvLMJ",
        "outputId": "e04a78e6-4b36-4477-d7d9-1d2a232f71ff"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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NrVXr14O233yyScwmUz257t27cITTzyB+++/Xz0/evSoCuKWLl1q38bTs+KHNTCoIyIiIiomOzsb8+bNwxdffIHmzZurx5EjRzBnzpwSQZ0EbDYFBQX44IMPMHLkSLRs2VItO378OOrWrYvIyEhUJo5+JSIiIk0MlKishzMHDx5Efn4+2rZta1/Wvn17lYUzm80ozcKFC1Vz66hRo+zLJFNXp04dVDYGdURERETFJCQkIDQ0FD4+PvZlERERqp9damoqnLFYLPjyyy8xZMgQ+Pv725cfO3YMsbGx6N+/P3r06IH//ve/iI+PR0VjUEdERESaGP1aWQ9ncnJyHAI6YXuel5fn9G9k4IMEbwMGDHBYLs2vmZmZGD9+vGqalYDu4YcfVk21FYl96oiIiIiK0ev1JYI323MZCevMypUrcc011zj0sRPLli2Dh4eH/e8+/vhjdO/eXTXltmvXDhWFmToiIiJyeZe7T110dDRSUlJUv7qiTbISmAUFBTn9m/Xr16Nnz54llvv6+joEguHh4Srwi4uLQ0ViUEdEREQu73IHdU2bNoW3tzd27txpX7Zt2zY1otXZdCTJyck4c+aMGkxRlDS7duzYEX/99Zd9mQRzEjDWq1evQsuIQR0RERGRk+xa37598eqrr2L37t1YvXo1vvrqKzUIwpa1y83NtW8v051Ik23NmjUdXicgIEAFejIJsbzOvn371EAJGTDRuHFjVCQGdUREROTyLvdACSEDG2R+uqFDh6o7S4wdOxY333yzWid94pYvXw6bpKQk1SwrfeeKe/vtt9GsWTM89NBD6u4TNWrUwLvvvouK5mGR8beX2TJdxUam7qRBv8qfx0arzPmlzwt0JQuIdt63gwBvXz2LoRRePhwnV5q004msN6VoOKcwiLncTj/cr9Jeu/aUhXAH3KuJiIjI5ZXW942qOKhjNqp0RxeevIzfhLY07F+3qt8CkdswZRur+i24LHN+xc4dRnS5MFNHRERELu9Sfd/IiiVERERE5AaYqSMiIiLX52RUKTlipo6IiIjIDTBTR0RERC6Po1/LxqCOiIiIXB4HSpSNza9EREREboCZOiIiInJ5bH4tGzN1RERERG6AmToiIiJyeexTVzZm6oiIiIjcADN1RERE5PLYp65szNQRERERuQFm6oiIiMjlMVNXNgZ1RERE5Po82bhYFpYQERER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VIulZOMkFpKpSqZNm2bvkibTorz22mv49NNPMXDgQAQHB1fKoNJy96mTiLR27cLbBG3cuBFeXl7qQxUd6isZPCIiIqKKdKnbeVUWX19fvP322+pR3KFDhXcFEa1atVLd0kojwZ48KlO5M3XSyU9mPhbSYvvbb7+ptmCJNm0kvegqd5cgIiIi91EV937VmnIHdXfeeadqA16zZo1qW5YOfvfff7/DzMtyb9jit84gIiIiospX7ubXRx55RHUEfP7551Wfuscffxy33367Widpya+//hrXXXed2o6IiIioIrlTRq3KgzoZ1isdBuXhbGJimc6kWbNmFf3+iIiIiKgcKuSuv40ba3fuNiIiInJ9zNRVYJ86IiIiInLzTB0RERGRO90mTIuYqSMiIiJyA8zUERERkctjn7qyMagjIiIil8egrmxsfiUiIiJyA8zUERERkctjpq5szNQRERERuQFm6oiIiMjlcUqTsjFTR0REROQGmKkjIiIil2cGJx8uCzN1RERERG6AmToiIiJyeRz9WjYGdUREROTyOFCibGx+JSIiInIDzNQRERGRy2Pza9mYqSMiIiJyA8zUERERkctjn7qyMVNHRERE5AaYqSMiIiKXxz51ZWOmjoiIiMgNVEmmzttXXxX/ljTO15PXIM40H9rksn8XWrHvm4NV/RZcVocnO1X1W3BZ3gafqn4L5AT71JWNza9ERETk8sxV/QY0gKkPIiIiIjfATB0RERG5PDa/lo2ZOiIiIiI3wEwdERERuTxOaVI2ZuqIiIiI3AAzdUREROTy2KeubMzUEREREbkBZuqIiIjI5bFPXdkY1BEREZHLM1uq+h24Pja/EhEREbkBZuqIiIjI5bH5tWzM1BERERG5AWbqiIiIyOVxSpOyMVNHRERE5AaYqSMiIiKXZ+Ho1zIxU0dERETkBpipIyIiIpdnhkdVvwWXx6COiIiIXB4HSpSNza9ERERE/4DFYsG7776LLl26oFOnTpg0aRLMZnOp2+/cuRP33Xcf2rZti1tuuQXz5s1zWN+nTx80btzY4XH48OFyvx9m6oiIiMjlueJAia+//hpLly7F5MmTkZ+fj2eeeQbh4eEYMWJEiW0TEhIwatQoDBw4EG+99Rb27duH8ePHIzIyEtdddx0KCgpw8uRJzJ49G3Xq1LH/XWhoaLnfD4M6IiIion9g5syZePzx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",
            "text/plain": [
              "<Figure size 800x600 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "plt.figure(figsize=(8,6))\n",
        "\n",
        "sns.heatmap(\n",
        "\tcorr_matrix,\n",
        "\tannot=True,\n",
        "\tcmap='coolwarm',\n",
        "\tvmin=-1,\n",
        "\tvmax=1,\n",
        "\tfmt='.2f'\n",
        ")\n",
        "\n",
        "plt.title('Correlation Matrix of Behavioral and Perception Constructs')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "a_YRbYU5vh2s",
      "metadata": {
        "id": "a_YRbYU5vh2s"
      },
      "source": [
        "#### Interpretation\n",
        "\n",
        "All constructs are positively correlated (**r** = **0.63–0.80**), indicating that higher scores in one construct are generally associated with higher scores in the others. The strongest relationships were **PU–FSC** (*r* = 0.80) and **PU–PEU** (*r* = 0.79), while **SP–IB** showed the weakest (*r* = 0.63). Overall, the constructs are closely interconnected."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "odW2qVXNL406",
      "metadata": {
        "id": "odW2qVXNL406"
      },
      "source": [
        "#### **4. Which of these constructs show the strongest relationship with AUB specifically?**\n",
        "\n",
        "Pearson correlation analysis was used to examine the relationship between each behavioral and perception construct and Actual Usage Behavior (`AUB`), allowing the strongest associations with social commerce usage to be identified."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "FdOo28znV_VI",
      "metadata": {
        "id": "FdOo28znV_VI"
      },
      "source": [
        "All behavioral and perception constructs are positively correlated with **Actual Usage Behavior (AUB)** (**r** = **0.66–0.79**). **PEU** shows the strongest relationship (*r* = 0.79), followed by **PU** (*r* = 0.77) and **FSC** (*r* = 0.74), while **TP** shows the weakest (*r* = 0.66). Overall, the positive correlations support the inclusion of all six constructs as predictor variables in the subsequent machine learning models."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "YaqJDX9SL7Tq",
      "metadata": {
        "id": "YaqJDX9SL7Tq"
      },
      "source": [
        "#### **5. How does Actual Usage Behavior vary across demographic groups?**\n",
        "\n",
        "Actual Usage Behavior (AUB) is compared across the demographic groups (`Gender`, `Income`, `Area`, and `Frequently`) to identify potential differences."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "kE9QWhLBr66r",
      "metadata": {
        "id": "kE9QWhLBr66r"
      },
      "source": [
        "#### A. AUB Across Genders\n",
        "\n",
        "Actual Usage Behavior (AUB) is compared between male and female respondents using descriptive statistics and boxplots. The **\"Different\"** category was excluded due to its small sample size (**n = 2**)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 33,
      "id": "Fsi9ZZFgVE1P",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 143
        },
        "id": "Fsi9ZZFgVE1P",
        "outputId": "04150f0f-1179-462f-8072-5bfe322ce2ca"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "      <th>mean</th>\n",
              "      <th>median</th>\n",
              "      <th>std</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Gender</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Male</th>\n",
              "      <td>167</td>\n",
              "      <td>3.658683</td>\n",
              "      <td>3.75</td>\n",
              "      <td>0.789311</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Female</th>\n",
              "      <td>588</td>\n",
              "      <td>3.686650</td>\n",
              "      <td>4.00</td>\n",
              "      <td>0.674685</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "        count      mean  median       std\n",
              "Gender                                   \n",
              "Male      167  3.658683    3.75  0.789311\n",
              "Female    588  3.686650    4.00  0.674685"
            ]
          },
          "execution_count": 33,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "gender_df = scommerce_df[scommerce_df['Gender'] != 3]\n",
        "\n",
        "gender_summary = (\n",
        "    gender_df\n",
        "    .groupby('Gender')['AUB']\n",
        "    .agg(['count', 'mean', 'median', 'std'])\n",
        ")\n",
        "\n",
        "gender_summary.index = gender_summary.index.map({\n",
        "    1: 'Male',\n",
        "    2: 'Female'\n",
        "})\n",
        "\n",
        "gender_summary"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "qZAeO2HPtuY7",
      "metadata": {
        "id": "qZAeO2HPtuY7"
      },
      "source": [
        "To better visualize the distribution of Actual Usage Behavior (AUB) across gender groups, a box plot is presented below."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 34,
      "id": "d_kW-IY5tt9M",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 465
        },
        "id": "d_kW-IY5tt9M",
        "outputId": "d63fd2e6-9ca3-4c16-ca72-915675a26dd4"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAi8AAAHACAYAAABqLoiOAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjksIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvJkbTWQAAAAlwSFlzAAAPYQAAD2EBqD+naQAAONhJREFUeJzt3Ql0FFXaxvE3CXvCYsiCEmQRCAiILCPigIyooAKiLOogoKKyCRlHEdkURtQIKDIQQBBGQRg2BRRFFJVBxwUEAooIwyayGRIjOySQ5Dvvnan+upNOCJKk+6b/v3P6VKq6uur2Uqmn71IdlJWVlSUAAACWCPZ1AQAAAC4G4QUAAFiF8AIAAKxCeAEAAFYhvAAAAKsQXgAAgFUILwAAwCqEFwAAYBXCC1DI/OE6kP5QBgAoKIQXBLRevXpJbGys61avXj1p0qSJdOnSRebOnSvnz5/3WL9t27YybNiwfG//008/laeffvqC6+k2ddu/dz+5OX78uAwdOlQ2bNjg8Zz15i/0Ndbnqq9706ZN5Ztvvsl13ZMnT0rjxo2lQYMGkpyc7HUdfR+nTJni9b4DBw6Y+5cuXeqxvvvt6quvlhtuuEGeeOIJOXjwYJ5l1+3oY3S7heWrr76Sv/zlL/KnP/1JGjZsKNdff73069dPvvjiCylK69atM89Vp4CvlfB1AQBf05PV6NGjzd8ZGRly7Ngx+fzzzyU+Pt6c9CdNmiTBwf/N+QkJCRIWFpbvbb/55pv5Wm/gwIHSu3dvKWg//vijvPvuu9K1a1fXMue5+gs9CS9btsy8Bhoa9P3Izfvvvy/ly5c379Pbb78tAwYMKJAydOvWTbp3727+PnfunAkt06dPlwcffFA++OADKVWqlPiCfgb1M3TrrbfKU089JdHR0Sa06Xv6yCOPmND30EMP+aRsgC8RXhDwNIxce+21Hq+D1nzUqlVLXnjhBXPCvPPOO83yvE6sl+LKK68ssvehdu3a4k+OHj1qplrbVa1atQvWdLRu3VpKliwpS5YsMTUQTrC8FFWqVPH4DPzhD38wyx544AFT86G1HkVNA50GF28B5fbbb5exY8fKK6+8IrfddptcfvnlRV4+wJdoNgJy0bNnT/NNd+HChbk25zjB5pprrjHV+UOGDJGkpCRznzbNrF+/3tyc6nan6l23edNNN5lmki+//DJHs5FTA/D888+bE2nz5s1N81NqamqezT/uVft6c2pzdOqsm/1xaWlpMnXqVHMSbNSokbRr105mzpwpmZmZHvsaOXKkWa4ncl3vvvvuk++++y7Pz4/WkMyfP186depkXiN97Msvv2z2qfR5O6/nLbfckmdz1q5du2TLli1mG/qaa+1IYTadVKxY0UyDgoIuuO6mTZvkrrvuMs06HTt2lJUrV7ru01ovfa2y01qdvGpN9D3R10zX8+axxx6TVq1ayW+//eYRBJ999llTg6Xv0T333CNff/21x+P086Hvib6f1113nWmu02aplJQUj/X0M9q+fXtTBj0WDh06lKMMukyb13Q72pynYW/btm05muneeOMN8/nSdd55551cnzOQX4QXILeDIzhYWrZsaU7Q2fu+qI0bN5r+JHqyf/3112X48OGmv8aTTz7pap7Rmhq9LVq0yPTTcGjzk4YRPdHoycObDz/8UH744Qd56aWXzLr/+te/5NFHHzWBID90f7p9pVNvzUXakbd///4ya9Ys02zy2muvmZOMNpVlX/+jjz4yfXhGjRolEydONCe7wYMH51ke3a82fWgw0WaY+++/X+bNm2eaiHTfOnWafvQ1yatJS096lSpVMqFPw1z16tVlwYIFUhA0qOl7rLf09HTZu3evqdXQ2jf9DFyIPk+tDZk2bZrUqVNH/vrXv8onn3ziapJKTEyUffv2udY/fPiwCZda2+TN9u3bZf/+/dKhQ4dcw1N4eLh5v5zaQA2EGh70PdL96+uptUfavJQ9wLz66qvmOev7qJ/hNWvWyIsvvui6X98jfS/atGljnpOGjmeeecZjGxqkNZTpZ1Tv09dLt6nv8e7duz3W1T5I+tkdP368/PGPf7zg6wlcCM1GQB4iIiJMDYh+o9W/s4eXMmXKSN++fV19IvTk+v3335sTszbPOP1jsjdL9ejRw4SEvFx22WUye/ZsKVeunGtev21rfxw9gV+I7ttpItKpt+Yi3ZY2i+hJTE+USk8u+rz+/ve/mxobPRkrPbFreZzndOrUKROqtF+N1jh4qynRfika5vQ1crYdFRVlTpi6bz05Ok1m9evXl5iYGK/PRff93nvvmVoN57W+++67zUlRg8ClNpvoCVpv7nQ/Gkrz099FQ9zDDz9s/r7xxhvlp59+MtvT0KZl1gCq/VTi4uLMOvp3aGio6cvijQYXVaNGDY/l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l8aNG8sTTzwh+/fvl969e5v5Hj16SFJSkmtbr732mjnmGzZsaLaZkJCQ674XLlxo1m3SpIlpUtqxYwdvs+UIL/A7devWlejoaPn8889dy1avXi233HKLx3ozZ840oUbbtletWiV33XWXjB07VlJSUnJsc968ebJixQp55ZVXTMCpXLmy9OnTR86dO1ckzwkIdPk5BidPniwvvfSSzJgxQz7++GPTV0ZvGj6Sk5Pl9ddfN+stX75c5syZIy+88II59h977DHzf+CHH37IsV+txdVg88wzz8iyZcukWbNmJhAdO3asSJ8/ChbhBX5b++I0HWkNy5dffmmWuatXr57553XttddKtWrVzDc5/Uf4008/5djerFmzZOjQodKiRQu56qqr5LnnnjP/vL744osie05AIBg9erSp4XC/nT59Ol/H4IMPPmhqWa6//nqpX7++3HDDDaYmRv9u166d7N2716x3+eWXm1rXli1bSkxMjAk42jy1c+fOHOXR/fbr109uuukmqVGjhjz++ONStWpVee+994r0dUHBosMu/JIGlbi4ODl//rx8/fXXpjZGv6m505oYDTX6TW3Pnj2ybds2szwjI8NjvVOnTskvv/wif/3rXyU4+P/z+tmzZ70GHQC/nx63GjTcZWZm5usY1C8hjjJlypiQ4T6vX2SUhpstW7aYWpzdu3fLjz/+aGpmdD/Z6f0TJkyQiRMnupalpaVx7FuO8AK/pFW7auPGjWaU0a233ppjnVdffVWWLFkiXbp0MU1G+o1P27Wzc8LM3//+d6lZs6bHfRUrViy05wAEIv2SUb16dY9lx48fz9cxGBIS4nGfe9Bxp8f9iy++KN27dzdB6emnnzZNQd7o8a8joLSWxl1YWNhFPjP4E5qN4JdKlCghbdq0MU1Ha9asydHfRWk7uLZjDxkyxAzNPHPmjFme/bdGK1SoYP6h6jcz/aeqN6121m9jTjU0gMJT0MegXvNJ+7loKNEvLpdddpn8+uuvOY59pWFJa32c/epNO/tu3ry5gJ4dfIHwAr9uOtJvWPpPz7062aHDLzXY6IiEDRs2mPZ05VQtu9O29EmTJpkwpNXUo0aNkk2bNkmtWrWK5LkAga4gj0ENK9qcrMFn69atpjlK+7t5O/Yfeugh07lXO/n+/PPPJjDpqCbtdwN70WwEv6XDH7XPi7daF6XVxmPGjJEOHTqY0UlahazVztr+feONN3qs+/DDD5u+L88++6wZgq3DK2fPnk2zEVBECvIY1BoXvXXu3Nl8udFOvWXLljXHfnZaK6sjEHUkk05r164t06dPN513Ya+gLG/1bAAAAH6KZiMAAGAVwgsAALAK4QUAAFiF8AIAAKxCeAEAAFYhvAAAAKsQXgAAgFUILwAKlf6isF5Z9bbbbpNrrrnG/Kqw/nift18ALgjr1q2T2NjYQtk2AP/AFXYBFBq9omqPHj1MgBk2bJjUq1dPfvvtN5k/f77cd9995pLt3n76AQDyQngBUGimTp1qfjBv5cqV5sf5VNWqVSU+Pl4OHz4sb775pvlxTQC4GDQbASgUmZmZsmzZMvPDeE5wcTd+/Hh56qmnzN/6w5pdunQxzUqdOnWSjz76yLWe1tho2Hn88celcePG5tfGtcbGob+T88QTT0iTJk2kffv28v3333vsR0NS//79zWPbtm0rCQkJkpGRYe5bunSpqQHSXyhu1qyZvPfee3waAAsQXgAUCv0F39TUVGnevLnX+6OioqRMmTKSnJws/fr1M+FlxYoV8sgjj5jAooHGoc1MDRo0kPfff1/atWsno0ePlhMnTpj79O89e/bIvHnzzC8Vv/HGG67H6U+3DRo0yPx4nwYpDUG6j9dee821TmJiovmxvsWLF5sfAwXg/2g2AlAotG+Lcv/V4K+++srUcjiuuOIKufXWW+WGG26Qnj17mmXVq1c3vw48Z84cV/DRDriPPvqo+fsvf/mLzJ0713T4rVOnjnz44YdmXsONGjhwoDz33HPm72+++UYOHTokS5YskeDgYKlVq5Y8/fTTMnz4cFc5goKCZMCAASZIAbAD4QVAoXCaio4fP+5apk07TpPPxx9/LAsWLDC1JmvWrDH3Oc6dOyc1a9Z0zdeoUcP1d1hYmJmeP39e9u7da5qAtCOwo1GjRq6/d+/eLUePHjVNQu7NWWfPnnWFK62VIbgAdiG8ACgUWoNSqVIl0yyjfVlU2bJlzXInNDghRPu5aL8Uj39OJf7/31PJkiVzbF+bhLwpVaqU62/dtta2TJs2Lcd65cuXN9PSpUv/zmcIwFfo8wKgUGj46Nq1q2n+0U612SUlJZmp1rDs27fPhBrn9umnn5q+KReiwUSDjXsn3W3btrn+1m1rs1F4eLhr2wcOHJDJkyeb5iIAdiK8ACg0gwcPlsjISDOiZ9WqVbJ//3757rvvzPBoDRDanKPXgdm6dau8+uqr8tNPP5nQMnHiRNMf5kK0Calz584yduxY2bJli7lAnY4mcmgHXB2araOaduzYYToB6761BigkJIR3HrAUzUYACo2GhLfeesvUvmjTjdawaLOONiNNmTJFbrnlFrOejv55+eWXZfbs2RIdHW1GG91555352oeGEQ0vOiRbOwf36tVLxo0bZ+7TgDJ9+nRz/z333CPlypUzV/rVTrsA7BWUlVvDMQAAgB+i2QgAAFiF8AIAAKxCeAEAAFYhvAAAAKsQXgAAgFUILwAAwCqEFwAAYBXCCwAAsArhBQAAWIXwAgAArEJ4AQAAViG8AAAAscn/AVvz+IZ1f+w9AAAAAElFTkSuQmCC",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "sns.boxplot(\n",
        "    data=gender_df,\n",
        "    x='Gender',\n",
        "    y='AUB'\n",
        ")\n",
        "\n",
        "plt.xticks([0, 1], ['Male', 'Female'])\n",
        "\n",
        "plt.title('Distribution of AUB by Gender')\n",
        "plt.xlabel('Gender')\n",
        "plt.ylabel('AUB')\n",
        "\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "NM91e8rxtqBM",
      "metadata": {
        "id": "NM91e8rxtqBM"
      },
      "source": [
        "Male and female respondents exhibit very similar AUB distributions. Although females reported slightly higher mean (M = 3.69, SD = 0.67) and median (4.00) scores than males (M = 3.66, SD = 0.79; Median = 3.75), the differences are minimal."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "IfPnJe4TLvkf",
      "metadata": {
        "id": "IfPnJe4TLvkf"
      },
      "source": [
        "Although descriptive statistics suggest only minor gender differences in AUB, an independent samples *t*-test was conducted to determine whether the mean AUB scores differ significantly. A significance level of α = 0.05 was used.\n",
        "\n",
        "- $H_0: \\mu_{Male} = \\mu_{Female}$\n",
        "- $H_1: \\mu_{Male} \\ne \\mu_{Female}$"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 35,
      "id": "aRphGz-u3dDw",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aRphGz-u3dDw",
        "outputId": "fd74cd34-d8d8-4a5d-85ef-a58ecded547c"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "T-statistic: -0.4167\n",
            "P-value: 0.6773\n"
          ]
        }
      ],
      "source": [
        "male = gender_df[gender_df['Gender'] == 1]['AUB']\n",
        "female = gender_df[gender_df['Gender'] == 2]['AUB']\n",
        "\n",
        "t_stat, p_value = stats.ttest_ind(\n",
        "    male,\n",
        "    female,\n",
        "    equal_var=False\n",
        ")\n",
        "\n",
        "print(f\"T-statistic: {t_stat:.4f}\")\n",
        "print(f\"P-value: {p_value:.4f}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "MLtUYdO23v_Y",
      "metadata": {
        "id": "MLtUYdO23v_Y"
      },
      "source": [
        "The independent samples *t*-test found no significant difference in Actual Usage Behavior (AUB) between male and female respondents (*t* = -0.417, *p* = 0.677). Therefore, the null hypothesis was not rejected."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "ny_bBweRB-td",
      "metadata": {
        "id": "ny_bBweRB-td"
      },
      "source": [
        "#### B. AUB Across Income\n",
        "\n",
        "Actual Usage Behavior (AUB) is compared across income groups. The distribution of respondents is first examined to identify any substantial imbalance in group sizes that may affect interpretation."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 36,
      "id": "hoPcwfUBFBkC",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 807
        },
        "id": "hoPcwfUBFBkC",
        "outputId": "acef6ed6-ea24-436f-8078-9c7add2addb3"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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",
            "text/plain": [
              "<Figure size 1500x800 with 5 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n",
        "\n",
        "axes = axes.flatten()\n",
        "\n",
        "for i, income in enumerate(sorted(scommerce_df['Income'].unique())):\n",
        "    sns.histplot(\n",
        "        scommerce_df[scommerce_df['Income'] == income]['AUB'],\n",
        "        kde=True,\n",
        "        ax=axes[i]\n",
        "    )\n",
        "\n",
        "    axes[i].set_title(f'Income {income}')\n",
        "    axes[i].set_xlabel('AUB')\n",
        "    axes[i].set_ylabel('Count')\n",
        "\n",
        "# Remove unused subplot\n",
        "fig.delaxes(axes[5])\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "WOmmIbpaHDCY",
      "metadata": {
        "id": "WOmmIbpaHDCY"
      },
      "source": [
        "The income groups are highly imbalanced, with very few respondents in the highest income categories. Combined with the non-normal distributions, particularly for the smallest groups, this violates the assumptions of one-way ANOVA. Therefore, only descriptive comparisons are presented."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "1SwLqolaICAe",
      "metadata": {
        "id": "1SwLqolaICAe"
      },
      "source": [
        "#### C. AUB Across Area\n",
        "\n",
        "Actual Usage Behavior (AUB) is compared across urban, suburban, and rural respondents to identify potential differences."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 37,
      "id": "KCj0iTRwNQHk",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 174
        },
        "id": "KCj0iTRwNQHk",
        "outputId": "8f4eaa65-ad49-4fd2-cef6-ee011b4662da"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "      <th>mean</th>\n",
              "      <th>median</th>\n",
              "      <th>std</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Area</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Urban</th>\n",
              "      <td>461</td>\n",
              "      <td>3.711497</td>\n",
              "      <td>4.00</td>\n",
              "      <td>0.700356</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Suburban</th>\n",
              "      <td>74</td>\n",
              "      <td>3.641892</td>\n",
              "      <td>3.75</td>\n",
              "      <td>0.691288</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Rural</th>\n",
              "      <td>222</td>\n",
              "      <td>3.615991</td>\n",
              "      <td>3.75</td>\n",
              "      <td>0.724337</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "          count      mean  median       std\n",
              "Area                                       \n",
              "Urban       461  3.711497    4.00  0.700356\n",
              "Suburban     74  3.641892    3.75  0.691288\n",
              "Rural       222  3.615991    3.75  0.724337"
            ]
          },
          "execution_count": 37,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "area_summary = (\n",
        "    scommerce_df\n",
        "    .groupby('Area')['AUB']\n",
        "    .agg(['count', 'mean', 'median', 'std'])\n",
        ")\n",
        "\n",
        "area_summary.index = area_summary.index.map({\n",
        "    1: 'Urban',\n",
        "    2: 'Suburban',\n",
        "    3: 'Rural'\n",
        "})\n",
        "\n",
        "area_summary"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "yQV8xnnFNbzh",
      "metadata": {
        "id": "yQV8xnnFNbzh"
      },
      "source": [
        "The table and boxplot summarize the distribution of AUB across residential areas."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 38,
      "id": "-7WfN3unNfnJ",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 465
        },
        "id": "-7WfN3unNfnJ",
        "outputId": "1cce64ac-1703-433c-896a-babd015fb328"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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Ww4gsFTtr6YZBRf1bSC4ALGBmhcAABAo1LwAAIBAIbwAAIBAIbwAAIBAIbwAAIBAKbI3qUtJyf8P0wEAgOhQufJ/716eF2peAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoPgaXhYtWmT169fP8jdw4MCI037yySfWqVMna9KkifXs2dO2bNlS6OsLAAD8F5eZmZnp18InTpxoq1atslGjRoXGlShRwsqWLZtluu3bt1vHjh1twIAB1qZNG3v++edt48aN9te//tXi4uIizpsfZgQAoGj+MKOvvyqtAFKvXj2rXLlyntO9/vrrdsEFF9idd97pHo8ePdp+9atf2dKlS61ly5aFtLYAAMBivdlI4aV27drHnU61My1atAg9LlWqlDVq1MhWrlx5mtcQAABEG99qXtRa9e2339q//vUvmzRpkh07dsyuuuoq1+elePHiWaZNSUmx5OTkLOMqVqxoP/zwQ57LyKVFKTB27vzBDhzY7/dqRIXExNJWpUpVv1cDlMsQymT0YF8Ze2XSt/CifiwHDx50QeXZZ5+1rVu32qOPPmqHDh2y4cOHZ5nWmy6cHqenp+c6/6SkREtICO7FVLt377YBA/pYRkaG36sSFeLj423u3LlWvnx5v1clplEu/4cyGR0ok7FZJn0LL9WqVbMlS5ZYuXLlXKfbhg0bugP1Qw89ZEOGDLGEhIQsnXizBxU9zt6xN1xa2oGA17wk2HPPTfK15mXr1i02fvw4GzjwfqtevYb5fUZx9GiCpabu83U9QLn0UCajBWWyqJXJSpWivMNu9nR47rnn2uHDh23Pnj2WlJQUGl+lShVLTU3NMq0eK/Dkxb/rqApGcrK/1X/e9qtWrYadc05d81vQP8+ignL5P5TJ6ECZjL0y6Vu7yscff+yuFFKTkOfrr792gSY8uIju7bJs2bLQY71mzZo1bjwAAIgtvoWXpk2buuYg9W/ZtGmTLV682J588knr1auX67yrTrpeU9GNN95oy5cvt8mTJ9v69etds1L16tW5TBoAgBjkW3gpXbq0TZ061dLS0lw4GTZsmN18880uvOzYscNat25tK1ascNMqqDz33HP25ptvWteuXV0HLd2oLrcb1AEAgKLL1z4v5513nr388ss5xiusrFu3Lsu4tm3buj8AABDbgnstMQAAiEmEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECjFLEr07t3bkpKS7Iknnoj4/LXXXmvr1q3LMm7BggVWr169QlpDAAAQDaIivLzzzju2ePFiu+GGGyI+f+zYMfvuu+9s5syZVrt27dD4ChUqFOJaAgCAaOB7eNm9e7c9+eST1rhx41yn2bp1qx05csQuvPBCK1GiRKGuHwAAiC6+h5cxY8bYddddZ7t27cp1mg0bNthZZ51FcAEAAP6Gl08//dQ+//xz13dl5MiRuU63ceNGO+OMM6xPnz62evVqO+ecc2zQoEGuJiYvcXGnYaVjiLf9NGRbIlpQLhFtKJMxFF4OHz5sI0aMsEceecRKliyZ57Tffvut7dmzx7p162YDBw60OXPm2O23324LFy50NTKRJCUlWkICF1OdirS0RDcsXz7RKlUqc0rzAgoK5RLRhjIZQ+FlwoQJdsEFF1ibNm2OO+2oUaPs0KFDVrp0afdYtTTLly+3t956y/r27RvxNWlpB6gtOEW7dx8IDVNT953q7IACQblEtKFMFqz8nCwX8/MKo9TUVGvatKl7nJ6e7obvv/++rVixIsu0xYoVCwUXiYuLszp16tjOnTvzXEZm5mlZ9ZjhbT8N2ZaIFpRLRBvKZOHzLbzMmDHDjh49Gnr81FNPueGDDz6YY9oePXpYy5YtrX///u5xRkaGu+fLbbfdVohrDAAAYjq8VKtWLcvjxMT/9q+oVauWu69LWlqalStXzooXL27t2rWz559/3ho2bOg6606fPt327duX631hAABA0eX7pdKR7Nixwy6//HIXUlTjcscdd7gOvo8++qhramrSpIm9/PLLWZqSAABAbIia8BL+swDVq1fP8lMA6uOijrm5dc4FAACxg2uJAQBAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABA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HjBdunSx0qVL26RJk2zEiBE2Z84cq1+/vk2ePJkb1J0mCii6HFpXFe3Ysd3OOuts11REjQsAIKbDi+6Yq7/cKMCEu/DCC23evHmFsGYQBZVevbgcGgAQfbiWGAAABArhBQAABArhBQAABArhBQAABArhBQAABArhBQAABArhBQAABArhBQAABArhBQAABIrvvyqN6KSfbli7do399FOaVaiQZA0anG8JCQl+rxYAAIQX5LRkySc2bdpUS0nZFRpXuXKy3X77Xday5S/ZZAAAX9FshBzB5emnn7CaNWvbY4+NtenT57ihHmu8ngcAwE+EF2RpKlKNS7NmF9mgQcOsXr0GVqpUKTfUY42fPv0vbjoAAPxCnxeEqI+Lmoruu+8hi4/Pmmv1+IYbutnw4Q+56Ro1asyWAxAVdu78wQ4c2O/b8rdt25Jl6JfExNJWpUpViwWEF4Soc67UqFEr4lapWbNmlukAwG979+6xAQP6WGZmht+rYuPHj/N1+fHx8fbSS9OtbNlyVtQRXhCiq4pky5bNrqkou++//z7LdADgNx2on3tukq81L9EiMbF0TAQXIbwgRJdD66qiuXNfd31cwpuOMjIybN681y05uYqbDgCiRaw0leB/6LCLEN3HRZdDL1/+mT355GO2bt1aO3jwZzfUY43v2fNO7vcCAPAVNS/IQvdxeeCBwe6qI3XO9ajGReO5zwsAwG+EF+SggNKiRUvusAsAiEqEF+TahMTl0ACAaESfFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAEChREV7S09OtU6dOtmTJklynueeee6x+/fpZ/j788MNCXU8AAOA/338e4PDhw/bAAw/Y+vXr85xu48aNNnbsWGvVqlVoXLly5QphDQEAQDTxNbxs2LDBBZfMzMzj1sxs3brVGjdubJUrVy609QMAANHH12ajpUuXWsuWLe21117Lc7pNmzZZXFyc1ahRo9DWDQAARCdfa15uvfXWfE2n8FK6dGkbNGiQCzxVq1a1AQMGWNu2bfN8XVxcAa0ogJCdO3+wAwf2+7ZFtm3bEhr6+R1PTCxtVapU9W8FgBjme5+X/IaXQ4cOWevWra137962aNEi14FXNTZqSookKSnREhKioj8yUGTs3r3bBgzoYxkZGX6vio0fP87X5cfH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qvx4fw0aNLCLL77Y7rnnHtuxY0eh7qeRt2LHeR4A8vTUU0/ZJ598Yo8++qjVqFHDtmzZYo899pht3rzZXnzxRbYeAmXo0KF2zTXXuP8zMjJsw4YNNmLECHv44Ydt+vTpfq8e/j/CC4BTMm/ePHv88cetVatW7nH16tVt5MiRdtttt9muXbssOTmZLYzAKFOmjFWuXDn0uEqVKjZw4EB76KGHbN++fe55+I9moxjmVdk///zzdtFFF9mf/vSnUDPAsGHDrEmTJnbFFVfYwoULQ69RU8CQIUPcgUpNBFdddZX9/e9/Dz2v+b311lvWqVMn9/ytt97qzsRRdMXFxdl//vMfd5bqadq0qb3zzjtWoUKFHNXiatJROQn33nvv2aWXXmrNmjWzRx55xNLT03Nt4lSzppqpZPDgwe7v2muvdWXyu+++c+M/++wza9++vSvDv/vd72zPnj2h13/wwQd2/fXXW+PGja1FixZ2//3324EDB9xzmu8DDzzgzrS1LprnSy+9dFq2G4KjePHibhgfH5+jWTK8jGq8/lf5ad68uU2ePNmVZTWjtmnTxho1auSef+2113x7L0UF4QW2fPlye/PNN61nz55ua6xYsSL0pfzNb35jDz74oGsCEDUHfPvtt/aXv/zF3n77bbfzV9DxDjbeAUDj9PqffvrJnn32WbZyEaZyM2PGjNBO+/3337dDhw5Z3bp17YwzzsjXPObMmWPPPPOMa2b66KOPbNKkSflevsLyfffd515Tu3ZtN27WrFmuDGqo8qqDh3z//fcuzChUv/vuu65sqslLy/do/UuUKOFqlO666y7XLKZ5IDapzCiEKHwkJiYed/pt27a5/aH2fzqJ02v/+c9/uv2iQrqC86hRoyw1NbVQ1r+oIrzAbr/9dqtZs2Zox69qflX7n3vuuW7nrTOI119/3T3n1dA0bNjQTX/nnXfa7t277ccffwxtyd/+9rfujLVevXou/KxevZqtXITde++9NnbsWKtataoLAapi145egfhE+hmonKlzpMLFq6++mu/XqgZFwenCCy8Mjevfv7+1bdvW1f4NHz7cFixY4GoNVTukxzfddJNr3mrdurX98pe/tPXr14deW758ede/oVatWtarVy/3mDIcOxTAVXOoP5UthQ3tC1XG80vlRuXn7LPPdp1+ddL3i1/8wvUJ69u3r6vd9moJcXLo81KEFStWLEtVvkfj9JynWrVqWZ5XMAk/Y1ZV58aNG93/+iKrmUgHqU2bNtlXX33lxh87diw0vb60ntKlS7svKoo2NdvoTzVt//rXv2zmzJmu5iN781BuwoPH+eef785Kw5t68pK9/IoOOuHzO3r0qDuD1v9qApg4caILLPpTh8zrrrsuNL1CTUJCQuixzrb1esQGhW81OaopUbUlqklRU6KaQPNLZcijpvd///vf9sQTT7h95po1a3LsM3HiqHkpwtSxLNLlqtk7namKPJzadbOHHS/MDBo0yMaMGWNly5Z1tSqRqvfz21SA4Fu7dq3bKXu0g+/cubNrRlJNjPrCZBdppx1e5ryfW1M5Un+a7LIHiezlV8LDR/j8tL4dO3Z0gUVNnjoj9q4syav88hNwsaNixYruBExB989//rMb169fv1xPwiKV5/AyqeZQdfbVCaNO/ujvUjAIL0WYznq9/ivhVq1a5b6YuQmvQpcvvvjC6tSp44KQ+rnoy6izkyuvvDJ0dszOPTZpx/3yyy+HziY9qt0oWbKkJSUluTDgdYiVSB24v/nmmyzlTcHnzDPPzPFalbP83BMm+/w0H50Nq3+Mmj6ffvpp1+9FNT7qz0X5RSQqx7oFwNdff22vvPKKG5ef8hxOTaB/+MMfXN9BBeWDBw+GyjJOHs1GRZhqRvSnKnJ9adSJctGiRfbhhx/mmf63b9/uOpRp564OZjow6QxEX+RSpUrZ3/72N3dQUidG7wql8A67iB1qUtRN6XRmqqp19RNQk486u6pMqPpdHWLfeOMNa9mypWtWUmfv7FTedJBQQB4/frzrayXqs6I+VarJ0XI0zE9zkgK2ApDKq+Z7yy23uP/Vf2XdunUu0Kj2Ud+DL7/80vVFACJRwO3atau98MILrmlUTZJqFtUJnZrT1THXuxopEpU57XNVlnfu3OluKyDsM08NNS9FmL5katb5+OOPXXWlgoyq8adMmeI6keVGHR11wLjhhhtcTYvCj+51oC+oOq3pagxVvau5QHee1D0RdGaC2KQrdtRnZMKECXb11Vdbnz59XAjRDl59nnQlkJoZu3Tp4ppp1CE3O5VNlSVNq3mpE7moU7g6z6oMqgzrbLVDhw7HXSd1GlefGw0VqHTW611mrY6Td9xxhwvnCurqcJy95ggI9/vf/97VuGj/p1oU7R91JZH2paqFzovCivaP2mfqNhO6vYQCEfvMUxOXSd0VAAAIEGpeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAEQF3alUP2nh/YI5AOSG8AIgKrzzzjtWs2ZN9/tDAJAXwgsA3/3444/26aefulv1f/7558f9sTsAsY3wAsB3+gFQ/VCifvguOTk5S+1Lu3bt3G/KtG7dOvT7RvrVaP1OkX4jRr91NGvWrND0ev7FF190r9OP4el1+t0lAEUHvyoNICqajPSr0fHx8S50zJ8/39XCxMXFuecXLFhgU6dOdcHk8OHDdvfdd7sfDtWvUW/atMn9WF5iYqILN3rttGnTbNy4ce7XovXDpCNHjrTLLrvM/Qo2gOCj5gWAr3bs2GHLly+3K664wj1u3769azZatmxZaBrVyKgzr34NXUGmYsWK7heo9avTCjt9+/a16dOnu2nPOussGz16tLVq1cqqV6/ufrFav3y+fv16394jgIJFzQsA32tdSpQo4Zp35OKLL7Zy5crZvHnzrEWLFm5ctWrVQtOrpmXt2rXWtGnT0Lhjx45ZQkKC+/+SSy6xVatW2dNPP20bN260r7/+2lJSUiwjI6PQ3xuA04PwAsD38HLo0CFr3rx5ljCifjBqDhKFG8/Ro0ddrcojjzwScX661Prxxx+3bt26uVqchx9+2Hr27FkI7wRAYSG8APDNt99+a2vWrLHhw4dby5YtQ+M3bNhgv//9723RokU5XnPOOefYBx984JqEvNoWdfD98ssv3Xxmz57t+sv06tXLPbd37153NZP6ywAoGujzAsDXWpfy5cvbzTffbPXq1Qv9XXPNNVa3bl3X+TY79X9RTY1qXtQstHjxYnvsscdcPxipUKGCu+xawWj16tUuBB05csTS09N9eIcATgfCCwBfw0vnzp2tePHiOZ5TR9tPPvnEdu7cmWV86dKl7aWXXrLvvvvOXV2k2pbbbrvN+vTp454fOnSo7d+/36677jobMGCA6+h75ZVXur4vAIqGuEzqUgEAQIBQ8wIAAAKF8AIAAAKF8AIAAAKF8AIAAAKF8AIAAAKF8AIAAAKF8AIAAAKF8AIAAAKF8AIAAAKF8AIAAAKF8AIAAAKF8AIAACxI/h8MllNmMI5GSQAAAABJRU5ErkJggg==",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "sns.boxplot(\n",
        "    data=scommerce_df,\n",
        "    x='Area',\n",
        "    y='AUB'\n",
        ")\n",
        "\n",
        "plt.xticks(\n",
        "    [0, 1, 2],\n",
        "    ['Urban', 'Suburban', 'Rural']\n",
        ")\n",
        "\n",
        "plt.title('Distribution of AUB by Area of Residence')\n",
        "plt.xlabel('Area')\n",
        "plt.ylabel('AUB')\n",
        "\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "Q8Su8AbKO_dF",
      "metadata": {
        "id": "Q8Su8AbKO_dF"
      },
      "source": [
        "Urban respondents reported the highest AUB (M = 3.71, SD = 0.70), while suburban (M = 3.64, SD = 0.69) and rural respondents (M = 3.62, SD = 0.72) showed similar values. Median AUB was 4.00 for urban respondents and 3.75 for the other two groups."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "kSis3lbwPHhP",
      "metadata": {
        "id": "kSis3lbwPHhP"
      },
      "source": [
        "Although descriptive statistics suggest only minor differences in AUB across residential areas, a one-way ANOVA was conducted to test whether the mean AUB scores differ significantly (α = 0.05).\n",
        "\n",
        "- $H_0: \\mu_{Urban} = \\mu_{Suburban} = \\mu_{Rural}$\n",
        "- $H_1:$ At least one group mean differs."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 39,
      "id": "wlsoJ-67PRTl",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wlsoJ-67PRTl",
        "outputId": "ff965afa-b4e2-4526-ef41-1026405bf35a"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "F-statistic: 1.4681\n",
            "P-value: 0.2310\n"
          ]
        }
      ],
      "source": [
        "urban = scommerce_df[scommerce_df['Area'] == 1]['AUB']\n",
        "suburban = scommerce_df[scommerce_df['Area'] == 2]['AUB']\n",
        "rural = scommerce_df[scommerce_df['Area'] == 3]['AUB']\n",
        "\n",
        "f_stat, p_value = stats.f_oneway(\n",
        "    urban,\n",
        "    suburban,\n",
        "    rural\n",
        ")\n",
        "\n",
        "print(f\"F-statistic: {f_stat:.4f}\")\n",
        "print(f\"P-value: {p_value:.4f}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "Iuzni7u9P_VW",
      "metadata": {
        "id": "Iuzni7u9P_VW"
      },
      "source": [
        "The one-way ANOVA found no significant differences in AUB across residential areas (*F* = 1.4681, *p* = 0.2310). Therefore, the null hypothesis was not rejected."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "Ym1ADl6GQPWM",
      "metadata": {
        "id": "Ym1ADl6GQPWM"
      },
      "source": [
        "#### D. AUB Across Frequency\n",
        "\n",
        "Actual Usage Behavior (AUB) is compared across social media usage frequency groups to identify potential differences."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 40,
      "id": "cQjBpYBVRGMF",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "cQjBpYBVRGMF",
        "outputId": "e73e3bb5-eb74-45e2-cdba-c96b883892df"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "      <th>mean</th>\n",
              "      <th>median</th>\n",
              "      <th>std</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Frequently</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Daily</th>\n",
              "      <td>725</td>\n",
              "      <td>3.681724</td>\n",
              "      <td>4.00</td>\n",
              "      <td>0.701587</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Weekly</th>\n",
              "      <td>14</td>\n",
              "      <td>3.678571</td>\n",
              "      <td>4.00</td>\n",
              "      <td>0.895655</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Monthly</th>\n",
              "      <td>7</td>\n",
              "      <td>3.750000</td>\n",
              "      <td>3.75</td>\n",
              "      <td>0.677003</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Rarely</th>\n",
              "      <td>11</td>\n",
              "      <td>3.295455</td>\n",
              "      <td>3.25</td>\n",
              "      <td>0.820200</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "            count      mean  median       std\n",
              "Frequently                                   \n",
              "Daily         725  3.681724    4.00  0.701587\n",
              "Weekly         14  3.678571    4.00  0.895655\n",
              "Monthly         7  3.750000    3.75  0.677003\n",
              "Rarely         11  3.295455    3.25  0.820200"
            ]
          },
          "execution_count": 40,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "frequency_summary = (\n",
        "    scommerce_df\n",
        "    .groupby('Frequently')['AUB']\n",
        "    .agg(['count', 'mean', 'median', 'std'])\n",
        ")\n",
        "\n",
        "frequency_summary.index = frequency_summary.index.map({\n",
        "    1: 'Daily',\n",
        "    2: 'Weekly',\n",
        "    3: 'Monthly',\n",
        "    4: 'Rarely'\n",
        "})\n",
        "\n",
        "frequency_summary"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "1bvalH9efWoK",
      "metadata": {
        "id": "1bvalH9efWoK"
      },
      "source": [
        "To better understand the distribution, a graph showing the distribution is shown below."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 41,
      "id": "qI9ytBd5faHv",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 807
        },
        "id": "qI9ytBd5faHv",
        "outputId": "c8654a10-4442-41a1-8696-2df960280bd3"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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BGORoBNEn5NNzX5/Ajl2ivCYIkch6P3339fDDz+s7777rum2NWvW6NBDD9XEiRPVvn37gG4fAABAsGDcBAAAglHI5mrHxcWptrZlEXDvdbOiDAAAABg3AQCA4BWyQal27dqppKTE1kdonppuAlKpqTteAQ8AAMBJGDcBAIBgFLJBqQEDBig6OlqzZs1qum369OkaMmQIxToBAAAYNwEAgCAXskGphIQEnXDCCbrttts0Z84cff311/rPf/6jc889N9CbBgAAEFQYNwEAgGAUskEp44YbbtCgQYP0+9//XrfffruuuOIKHX744T59T1O36phjjtGUKVO2+5gFCxbo1FNP1bBhw3TyySdr3rx5ClWtae8ll1yifv36tTg1L0AfCvLz83XllVdqzJgx2n///XXPPffYFYnCef/uSpvDYR+vWrVKF1xwgV0Offz48Xr22We3+9hw2ce70uZw2MfNXXTRRbr++uu3e//kyZPtb5vZx+ZghlkoI5TtrL3HHXfcVvt3yZIlCjVfffXVVu0wv2NO2MehOm4KJk4Z07SGE8c9O+O0cVFrOHHstDNOHlu1htPGX63hlDHaHjGr76F1ampqPJdddpmnb9++nl9++WWbj6msrPSMGzfO849//MOzdOlSz5133unZd9997e3h2F7jsMMO83zwwQeeTZs2NZ1cLpcnVLjdbs9pp53mufDCCz1LlizxTJs2zbbJ7MNw3b+70uZw2McNDQ2eww8/3PPXv/7Vs2LFCs/333/vGTlypOfDDz8M2328K20Oh33c3Mcff2x/t6677rpt3r9u3TrP8OHDPc8995z9/P/5z3/2HHPMMfZ7EY7tra+v9wwZMsQzderUFvu3rq7OE2qeeOIJz8UXX9yiHWVlZWG/j7HnnDKmaQ0njnt2xmnjotZw4thpZ5w8tmoNp42/WsNJY7Q9QVCqlfLy8jzHHXec59hjj93hgOatt97yHHzwwU1fLnNufpDeeecdTzi21/ywDhgwwLN8+XJPqDL/eZo2FhQUNN320Ucfefbbb7+w3b+70uZw2Mf5+fn2P77y8vKm28wfJ7feemvY7uNdaXM47GOvkpISzwEHHOA5+eSTtzsAeOihhzxnn3120/WqqirPiBEjdviHaii3d+XKlZ7+/fvbP8pDnflD4F//+tdOHxdO+xh7zkljmtZw4rhnZ5w2LmoNJ46d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9dfb0/a27eCDD97mfeZ2c38wefTRR21fek9DhgzR8ccfv0t9taP+2BNr166122TO29Jf//pXTZ48udWfqe2177fbd84557ToyxEjRuiCCy7QqlWrmp7z5ptv6sEHH2zT9gAAsDOMndoOY6edf6YYOwG7jqAUEEI++eQTde3aVR988EGrn3PjjTdqzpw5Pt2uUGWCJ5MmTbIn07fHHHOMrrjiijYPBu2qDh062G0y523ll19+UX5+vvbdd989/kxty/nnn2+3+ccff7QBqPT0dF166aXyeDz2/pNOOklffvmlVqxYsUfvAwDArmDs1LYYOzF2AtoaQSkgRBQVFennn3/WZZddpl9//dVmQGHPxMTEKCcnx55MYOaPf/yjOnbsqG+//TagXRsVFWW3yZy3lSeeeEK/+93vfPaZSkxMtNucm5urPn362COFS5cu1eLFi+390dHROvHEE/Xvf/97j9sCAEBrMHZqe4ydGDsBbY2gFBAiPv/8c6WkpOi4446zf/g3z2ypqqrSLbfcorFjx9rT3/72N7lcLhsYmDp1qh577DE7xWpb08JMKra5z8ukIk+YMEGDBw+2r3X77beroaGhzdqxfv16m1VjjrTts88+uvPOO1VXV2fvq6io0A033GBvN+9vtuPrr79uem5JSYkuv/xy+9xDDjlEr732mm2P15IlS2xbhg4dqiOOOEKvvPLKbgVXmjPBGpPlY17z2GOP1RdffNHifrPNV111lYYNG6bx48fro48+arrPZCZdeeWV2muvvWx7TFBm+vTp9j7znOuuu26r6XU33XTTVvuprKzM7lOT5TRq1Chdc8019jZjypQpdnrkrbfeau975plntmrT8uXLNWPGDB144IGt/kztqYSEhK1uM/vMHLHevHlzm70PAADbw9iJsRNjJyD4EZQCQoT5Y94EPSIjI20Q4v3332+aGnXzzTfbYIfJhvnPf/5jLz/00EM2wGECOCYIZIJPO2MCWHfddZf+8pe/2IGcCUi9/fbb+uabb9qsHSYIZQI/Zvsff/xxG+Qx072Mu+++207vMm34+OOPNXr0aNuG2tpae7/ZruLiYhuMMkE483yvmpoam+lkAjMffvihDfiY/jDv0xqmL00AbOXKlTrssMPsbQUFBbr44ottUMoEmy688EIb6DOBKq+vvvpKgwYNstt75JFH2umS5eXl9r6r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04BIPMcXXXTRAce0cePGnmzbH//4xzrppJM0Z84c/ehHP1JXV5dC3eGOef78+Qd9Hd999909n1v732efp6FmyZIlB+znV77yFc+8jsOt79RbWVmZO1f2mrJ2v+OOO9yKysfSj/Kqo2nDD+qneFFf9uW9qi/7yl53vH1RHF9fz8uW9GFfqt/4cYDOzk7/ggUL/F//+tf9W7du9b/88sv+WbNm+f/xj38c8Nimpib/qaee6v/hD3/oLy4u9t92223+U045xd1uVqxY4Z82bZr/scce869bt87/8Y9/3H/NNdeE7fHaMUyePNn/wAMP+Ldt2+b/05/+5H62201paal/3Lhx/h07dvjLy8t7tq6uLn+4HrM555xz/E888cQ+x9TW1ubue+aZZ/yzZ8/2v/jii+58n3/++f5bb73VH67HW1NTs89xLlmyxJ3jlStXuvvfe+89/5w5c/Z5TGVlpT9UPfXUU+45ecMNNxz0/t27d/tnzJjh/93vfuffuHGj/7/+67/8ixYt6nnOhsP5PdpjfuWVV9z7kj2n7XV81113ueeDvX4j8Rx3dHT4p06d6n/nnXf2Oab29nZ3v537M844w//uu+/633zzTf+8efP89913nz+UfdAxV1VV7XOsf/zjH93zeNeuXe5+O/cXX3zxPo+prq72h5p77rnHf+211+6zn3V1dZ54HYcza/errrrK/9nPftadD3tt2eeo9ZWOth/lVUfThh/UT/GivuzLe1Vf9pW97nj7ojj+vp6X3dNHfan+RFDqIMrKytxJaGho6LntS1/6kv+WW2454LGPPPKIf/78+T0nyy7tTfnvf/+7+/mb3/zmPi+ekpIS//jx413QJhyP98477/R/5jOf2ee2q6++2v/Tn/7UXX/jjTfcB3uoO5pjtg/UiRMn+rds2XLQ3/XRj37U//Of/7znZ+s42hf+5uZmfzge7/5v8PblzYIWAQ8//LD/wx/+sD8cWIDt9NNP919++eWH/BD72c9+5oLFAXbeZs6c6X/rrbfC5vwe7TH/93//t/873/nOPrdZx/Ohhx6KyHNsgbcJEyb4W1tbD3q/BaQC79nm8ccf95911ln+cD7m3urr6/0nnXSSO68B9p79ta99zR/q7MvQT37ykw98XKS9jsOdfbG3Lw8VFQz0KRgAABfNSURBVBU9tz355JMu4Hu0/SivOpo2/KB+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",
            "text/plain": [
              "<Figure size 1200x800 with 4 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Label mapping\n",
        "frequency_labels = {\n",
        "    1: 'Daily',\n",
        "    2: 'Weekly',\n",
        "    3: 'Monthly',\n",
        "    4: 'Rarely'\n",
        "}\n",
        "\n",
        "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
        "axes = axes.flatten()\n",
        "\n",
        "for i, freq in enumerate(sorted(scommerce_df['Frequently'].unique())):\n",
        "    sns.histplot(\n",
        "        scommerce_df[scommerce_df['Frequently'] == freq]['AUB'],\n",
        "        kde=True,\n",
        "        ax=axes[i]\n",
        "    )\n",
        "\n",
        "    axes[i].set_title(frequency_labels[freq])\n",
        "    axes[i].set_xlabel('Actual Usage Behavior (AUB)')\n",
        "    axes[i].set_ylabel('Count')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "JPLzWxh3RIGN",
      "metadata": {
        "id": "JPLzWxh3RIGN"
      },
      "source": [
        "The frequency groups are highly imbalanced, with very few respondents in the weekly, monthly, and rarely used categories. Therefore, the analysis is limited to descriptive comparisons of AUB across frequency groups."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "7odvhZst1W5i",
      "metadata": {
        "id": "7odvhZst1W5i"
      },
      "source": [
        "### [4] Research Question"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "_-TtX9bgYw2I",
      "metadata": {
        "id": "_-TtX9bgYw2I"
      },
      "source": [
        "#### **How does actual usage behavior differ among clusters of Generation Z university students formed based on their perceived usefulness, perceived ease of use, familiarity with social commerce, social presence, trust in platform, and interaction behavior?**"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "7X-Wfl7FY5V6",
      "metadata": {
        "id": "7X-Wfl7FY5V6"
      },
      "source": [
        "The exploratory data analysis showed that the behavioral and perception constructs are moderately to strongly correlated, suggesting that respondents may form distinct user segments.\n",
        "\n",
        "This study investigates whether clustering respondents based on these constructs reveals differences in Actual Usage Behavior (AUB). The results provide insights into user segments that may support targeted marketing, personalized platform improvements, and future social commerce research."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "VVykRdx_1GRo",
      "metadata": {
        "id": "VVykRdx_1GRo"
      },
      "source": [
        "## **CBDATSI Phase 2**\n",
        "\n",
        "The first phase of the case study involves four sections – (1) Data Modelling, (2) Statistical Inference, (3) Insights and Conclusions."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "nzqrUNibD8vm",
      "metadata": {
        "id": "nzqrUNibD8vm"
      },
      "source": [
        "### [1] Data Modelling\n",
        "\n",
        "#### **Feature Selection**\n",
        "The six behavioral constructs were used as features for clustering because they represent the respondents' perceptions and behaviors associated with social commerce.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 42,
      "id": "3QiIV3Gx2vmk",
      "metadata": {
        "id": "3QiIV3Gx2vmk"
      },
      "outputs": [],
      "source": [
        "features = [\n",
        "    \"PU\",\n",
        "    \"PEU\",\n",
        "    \"FSC\",\n",
        "    \"SP\",\n",
        "    \"TP\",\n",
        "    \"IB\"\n",
        "]\n",
        "\n",
        "X = scommerce_df[features]"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "-utoRfjM_WqN",
      "metadata": {
        "id": "-utoRfjM_WqN"
      },
      "source": [
        " #### **Feature Standardization**\n",
        " The selected constructs were standardized to ensure equal contribution to the distance calculations used by K-Means."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 43,
      "id": "tj-NqLRJ3hDp",
      "metadata": {
        "id": "tj-NqLRJ3hDp"
      },
      "outputs": [],
      "source": [
        "scaler = StandardScaler()\n",
        "\n",
        "X_scaled = scaler.fit_transform(X)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "7ZrOR4V8cjGM",
      "metadata": {
        "id": "7ZrOR4V8cjGM"
      },
      "source": [
        "#### **Variance Inflation Factor (VIF)**\n",
        "VIF was computed to assess multicollinearity before clustering, with VIF > 5 indicating potentially problematic multicollinearity."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 44,
      "id": "7HQM8mcCX_nG",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7HQM8mcCX_nG",
        "outputId": "0a1ff3df-e9c1-4c36-d2d0-c8b002c9faf8"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "  Construct       VIF\n",
            "0        PU  3.654939\n",
            "1       PEU  3.515319\n",
            "2       FSC  3.988501\n",
            "3        SP  3.150629\n",
            "4        TP  2.709242\n",
            "5        IB  2.301402\n"
          ]
        }
      ],
      "source": [
        "vif_data = pd.DataFrame()\n",
        "vif_data[\"Construct\"] = features\n",
        "vif_data[\"VIF\"] = [\n",
        "    variance_inflation_factor(X_scaled, i)\n",
        "    for i in range(X_scaled.shape[1])\n",
        "]\n",
        "\n",
        "print(vif_data)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "rs1UsVTPd68o",
      "metadata": {
        "id": "rs1UsVTPd68o"
      },
      "source": [
        "VIF values ranged from 2.30 to 3.99, indicating no severe multicollinearity. Therefore, all constructs were retained."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "phou4byrAl9K",
      "metadata": {
        "id": "phou4byrAl9K"
      },
      "source": [
        "#### **Clustering Algorithm**\n",
        "K-Means was selected because of its computational efficiency and scalability for clustering the dataset."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "VnAKLGHuAStV",
      "metadata": {
        "id": "VnAKLGHuAStV"
      },
      "source": [
        "#### **Selecting Optimal Number of k**\n",
        "The optimal number of clusters was determined using the Elbow Method and Silhouette Analysis. The optimal value of k was selected by considering the results of both methods."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 45,
      "id": "T6fMO43WAVnL",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 507
        },
        "id": "T6fMO43WAVnL",
        "outputId": "cbf633bd-09e0-421b-c050-f00428d64d58"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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",
            "text/plain": [
              "<Figure size 1400x500 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "k_values = range(2, 11)\n",
        "\n",
        "inertia = []\n",
        "silhouette_scores = []\n",
        "\n",
        "for k in k_values:\n",
        "    kmeans = KMeans(\n",
        "        n_clusters=k,\n",
        "        random_state=42,\n",
        "        n_init=10\n",
        "    )\n",
        "\n",
        "    labels = kmeans.fit_predict(X_scaled)\n",
        "\n",
        "    inertia.append(kmeans.inertia_)\n",
        "    silhouette_scores.append(\n",
        "        silhouette_score(X_scaled, labels)\n",
        "    )\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14,5))\n",
        "\n",
        "# Elbow Method\n",
        "axes[0].plot(k_values, inertia, marker=\"o\")\n",
        "axes[0].set_title(\"Elbow Method\")\n",
        "axes[0].set_xlabel(\"Number of Clusters (k)\")\n",
        "axes[0].set_ylabel(\"Inertia\")\n",
        "axes[0].grid(True)\n",
        "\n",
        "# Silhouette Score\n",
        "axes[1].plot(k_values, silhouette_scores, marker=\"o\")\n",
        "axes[1].set_title(\"Silhouette Score\")\n",
        "axes[1].set_xlabel(\"Number of Clusters (k)\")\n",
        "axes[1].set_ylabel(\"Silhouette Score\")\n",
        "axes[1].grid(True)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "0jKf-E16Emjc",
      "metadata": {
        "id": "0jKf-E16Emjc"
      },
      "source": [
        "Based on the Elbow Method and Silhouette Analysis, k = 4 was selected as the optimal number of clusters. The elbow plot shows diminishing returns beyond k=4, while the silhouette score indicates that k=4 provides the best clustering solution."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "NoZmt0qUQvDY",
      "metadata": {
        "id": "NoZmt0qUQvDY"
      },
      "source": [
        "#### **K-Means Clustering**\n",
        "\n",
        "Now that the optimal number of clusters is determined, we can now start clustering."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 46,
      "id": "KLGZYKp7AvYv",
      "metadata": {
        "id": "KLGZYKp7AvYv"
      },
      "outputs": [],
      "source": [
        "kmeans = KMeans(\n",
        "    n_clusters=4,\n",
        "    random_state=42\n",
        ")\n",
        "\n",
        "clusters = kmeans.fit_predict(X_scaled)\n",
        "\n",
        "scommerce_df[\"Cluster\"] = clusters"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "IlLhs1O_KoRf",
      "metadata": {
        "id": "IlLhs1O_KoRf"
      },
      "source": [
        "#### **Cluster Size**\n",
        "Cluster sizes were examined to describe the distribution of respondents across the identified clusters."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 47,
      "id": "rEsE1XkoAyKK",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 240
        },
        "id": "rEsE1XkoAyKK",
        "outputId": "a654680d-493e-4339-8b81-654a6d5cd769"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Cluster\n",
              "0     97\n",
              "1    258\n",
              "2     20\n",
              "3    382\n",
              "Name: count, dtype: int64"
            ]
          },
          "execution_count": 47,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "cluster_sizes = (\n",
        "    scommerce_df[\"Cluster\"]\n",
        "    .value_counts()\n",
        "    .sort_index()\n",
        ")\n",
        "\n",
        "cluster_sizes"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "1PkerzdYLaWv",
      "metadata": {
        "id": "1PkerzdYLaWv"
      },
      "source": [
        "Cluster 3 contained the largest number of respondents (382), followed by Cluster 1 (258), Cluster 0 (97), and Cluster 2 (20)."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "doawt2yPQMre",
      "metadata": {
        "id": "doawt2yPQMre"
      },
      "source": [
        "#### **Cluster Profiling**\n",
        "Cluster profiles were generated by computing the mean values of the six behavioral constructs together with AUB for each cluster. These mean scores were used to characterize the identified user segments."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 48,
      "id": "NA3kUFtuA1Bn",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 442
        },
        "id": "NA3kUFtuA1Bn",
        "outputId": "d1a02cc1-3047-487f-b741-52ac1e09c988"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: ylabel='Cluster'>"
            ]
          },
          "execution_count": 48,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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",
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "cluster_profile = (\n",
        "    scommerce_df\n",
        "    .groupby(\"Cluster\")[features + [\"AUB\"]]\n",
        "    .mean()\n",
        ")\n",
        "\n",
        "sns.heatmap(\n",
        "    cluster_profile,\n",
        "    annot=True,\n",
        "    cmap=\"Blues\"\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "yccxyHA6shOV",
      "metadata": {
        "id": "yccxyHA6shOV"
      },
      "source": [
        "#### **Cluster 0: Highly Engaged Social Commerce Users**\n",
        "\n",
        "AUB = 4.47, PU = 4.64, PEU = 4.58, FSC = 4.71, SP = 4.58, TP = 4.47, IB = 4.53\n",
        "\n",
        "Cluster 0 consists of respondents who exhibit consistently high scores across all behavioral and perception constructs, with mean values above 4.4. Respondents in this cluster perceive social commerce platforms as useful and easy to use, are highly familiar with social commerce, trust the platforms, experience a strong sense of social presence, and actively interact with others. Consequently, they also demonstrate a high level of Actual Usage Behavior (AUB = 4.47). These respondents can be characterized as highly engaged and active social commerce users, indicating that positive perceptions across all measured constructs are associated with frequent use of social commerce platforms."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "TwZ83wGotBLW",
      "metadata": {
        "id": "TwZ83wGotBLW"
      },
      "source": [
        "#### **Cluster 1: Moderate or Occasional Users**\n",
        "\n",
        "AUB = 3.17, PU = 3.25, PEU = 3.16, FSC = 3.10, SP = 3.05, TP = 3.03, IB = 3.17\n",
        "\n",
        "Cluster 1 represents respondents with generally neutral to moderately positive perceptions of social commerce. Their scores across all constructs are close to 3, suggesting that they neither strongly agree nor disagree with the statements measuring the constructs. This is reflected in their moderate Actual Usage Behavior (AUB = 3.17). These respondents may use social commerce occasionally but are not highly engaged. Improving their perceptions of the platform, particularly in areas such as trust and social interaction, may encourage greater adoption and more frequent usage."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "AZvZpAKLtDFu",
      "metadata": {
        "id": "AZvZpAKLtDFu"
      },
      "source": [
        "#### **Cluster 2: Low Engagement Users**\n",
        "\n",
        "AUB = 1.71, PU = 1.65, PEU = 1.57, FSC = 1.58, SP = 1.78, TP = 1.47, IB = 1.66\n",
        "\n",
        "Cluster 2 contains the smallest number of respondents and exhibits the lowest mean scores across all constructs. Respondents in this cluster generally disagree that social commerce platforms are useful, easy to use, trustworthy, or socially engaging, resulting in the lowest level of Actual Usage Behavior (AUB = 1.71) among all clusters."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "Owzn0HtotD3Z",
      "metadata": {
        "id": "Owzn0HtotD3Z"
      },
      "source": [
        "#### **Cluster 3: Regular Users with Positive Perceptions**\n",
        "\n",
        "AUB = 3.92, PU = 4.02, PEU = 3.96, FSC = 4.02, SP = 3.82, TP = 3.79, IB = 3.77\n",
        "\n",
        "Cluster 3 is the largest cluster in the dataset and represents respondents with generally positive perceptions of social commerce. While their scores are slightly lower than those of Cluster 0, they still report favorable levels of perceived usefulness, ease of use, familiarity, trust, social presence, and interaction behavior, which is reflected in their relatively high Actual Usage Behavior (AUB = 3.92). Compared with Cluster 1, respondents in this cluster consistently exhibit higher scores across all constructs, suggesting stronger confidence and engagement with social commerce rather than the moderate or uncertain perceptions observed in Cluster 1. However, compared to Cluster 0, they appear to be regular users rather than highly engaged users. Since this is the largest cluster, it may represent the typical Generation Z university student who has adopted social commerce and uses it regularly but whose engagement and confidence are not as strong as those observed in Cluster 0."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "5v0cHKsPbpff",
      "metadata": {
        "id": "5v0cHKsPbpff"
      },
      "source": [
        "#### **Table Summary**\n",
        "\n",
        "Below is the summary of Overall Profile, AUB, and Main Characteristics of the four clusters.\n",
        "\n",
        "$$\n",
        "\\begin{array}{|l|l|c|l|}\n",
        "\\hline\n",
        "\\textbf{Cluster} & \\textbf{Overall Profile} & \\textbf{AUB} & \\textbf{Main Characteristic} \\\\\n",
        "\\hline\n",
        "0 & \\text{Highly Engaged Users} & 4.47 & \\text{Very high engagement across all constructs} \\\\\n",
        "\\hline\n",
        "1 & \\text{Moderate/Occasional Users} & 3.17 & \\text{Neutral perceptions and moderate usage} \\\\\n",
        "\\hline\n",
        "2 & \\text{Low Engagement Users} & 1.71 & \\text{Low perceptions and minimal usage} \\\\\n",
        "\\hline\n",
        "3 & \\text{Regular Users} & 3.92 & \\text{Positive perceptions and regular usage} \\\\\n",
        "\\hline\n",
        "\\end{array}\n",
        "$$"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "5NzxWOl0IZB1",
      "metadata": {
        "id": "5NzxWOl0IZB1"
      },
      "source": [
        "### [2] Statistical Inference"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "XMpgbSKXTjHB",
      "metadata": {
        "id": "XMpgbSKXTjHB"
      },
      "source": [
        "Since AUB was excluded during cluster formation, it was analyzed separately to determine whether actual usage behavior differs significantly across the identified clusters."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "TDk7CEZOXISt",
      "metadata": {
        "id": "TDk7CEZOXISt"
      },
      "source": [
        "To determine whether AUB differs among the identified clusters, the following hypotheses are formulated.\n",
        "\n",
        "\n",
        "$$\n",
        "H_0:\\ \\mu_0 = \\mu_1 = \\mu_2 = \\mu_3,\\ \\text{where the mean AUB is equal across all clusters.}\n",
        "$$\n",
        "\n",
        "$$\n",
        "H_1:\\ \\exists\\, i \\ne j : \\mu_i \\ne \\mu_j,\\ \\text{where at least one cluster has a different mean AUB.}\n",
        "$$"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "cfoIPJeCUmZc",
      "metadata": {
        "id": "cfoIPJeCUmZc"
      },
      "source": [
        "#### **Assumption Checking**\n",
        "\n",
        "Before performing One-Way ANOVA, its assumptions were evaluated. Independent observations and a continuous dependent variable were satisfied; therefore, only normality was assessed using the Shapiro-Wilk test. The hypotheses are defined as follows.\n",
        "\n",
        "$H_0$: The AUB values within each cluster are normally distributed.\n",
        "\n",
        "$H_1$: The AUB values within each cluster are not normally distributed."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 49,
      "id": "9UjFotBVUqjZ",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9UjFotBVUqjZ",
        "outputId": "706b1b40-be06-450e-c857-9b2acbf8f00d"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Cluster 0: W=0.8871, p=0.0000\n",
            "Cluster 1: W=0.8512, p=0.0000\n",
            "Cluster 2: W=0.8787, p=0.0168\n",
            "Cluster 3: W=0.8786, p=0.0000\n"
          ]
        }
      ],
      "source": [
        "for cluster in sorted(scommerce_df[\"Cluster\"].unique()):\n",
        "    stat, p = shapiro(\n",
        "        scommerce_df.loc[scommerce_df[\"Cluster\"] == cluster, \"AUB\"]\n",
        "    )\n",
        "\n",
        "    print(f\"Cluster {cluster}: W={stat:.4f}, p={p:.4f}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "SClRks3bVI7J",
      "metadata": {
        "id": "SClRks3bVI7J"
      },
      "source": [
        "Since all clusters yielded p < .05, the normality assumption was violated. Therefore, the Kruskal–Wallis H test was used instead of One-Way ANOVA, and the hypotheses were reformulated as follows."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "yuB2GnG4XhIl",
      "metadata": {
        "id": "yuB2GnG4XhIl"
      },
      "source": [
        "### **The hypotheses for the Kruskal–Wallis H test are formulated as follows.**\n",
        "\n",
        "$$\n",
        "H_0:\\ \\text{The distribution of AUB is identical across all clusters.}\n",
        "$$\n",
        "\n",
        "$$\n",
        "H_1:\\ \\text{At least one cluster has a different distribution of AUB.}\n",
        "$$"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "5kRzr6ZHXtAt",
      "metadata": {
        "id": "5kRzr6ZHXtAt"
      },
      "source": [
        "The Kruskal–Wallis H test compares the distributions of groups using ranked observations and does not require the assumption of normality. Therefore, it is an appropriate non-parametric alternative to One-Way ANOVA for comparing AUB across clusters.\n",
        "\n",
        "The assumptions of independent observations, independent groups, and an ordinal/continuous dependent variable were satisfied because each respondent belonged to only one cluster and AUB was treated as a continuous aggregated Likert-scale score. The test was therefore performed at $\\alpha = 0.05$."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 50,
      "id": "HKyPA01n9lM3",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HKyPA01n9lM3",
        "outputId": "e2f23212-7e79-4db5-aa01-5d2153dc3c81"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "H-statistic: 421.3159298947122\n",
            "p-value: 5.341699041631374e-91\n"
          ]
        }
      ],
      "source": [
        "# Extract the AUB scores for each cluster.\n",
        "# Each element in 'groups' contains the AUB values of one cluster.\n",
        "groups = [\n",
        "    scommerce_df.loc[scommerce_df[\"Cluster\"] == c, \"AUB\"]\n",
        "    for c in sorted(scommerce_df[\"Cluster\"].unique())\n",
        "]\n",
        "\n",
        "# Perform the Kruskal–Wallis H test.\n",
        "# The * operator unpacks the list so that each cluster is passed\n",
        "# as a separate sample to the function.\n",
        "H, p = kruskal(*groups)\n",
        "\n",
        "# Display the test statistic (H) and the corresponding p-value.\n",
        "# H measures the difference in the average ranks of the groups,\n",
        "# while the p-value determines whether the observed differences\n",
        "# are statistically significant.\n",
        "print(f\"H-statistic: {H}\")\n",
        "print(f\"p-value: {p}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "0kVV4Clg1DxR",
      "metadata": {
        "id": "0kVV4Clg1DxR"
      },
      "source": [
        "The Kruskal–Wallis H test revealed a statistically significant difference in the distribution of AUB across the four identified clusters (H = 421.32, p < .001). Therefore, the null hypothesis was rejected. Dunn's post hoc test with Bonferroni correction was subsequently performed to identify which cluster pairs differed significantly."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "KZEu09QPX6pF",
      "metadata": {
        "id": "KZEu09QPX6pF"
      },
      "source": [
        "#### **Post Hoc Analysis**\n",
        "\n",
        "Dunn's post hoc test with Bonferroni correction was performed to identify the significantly different cluster pairs. For each pairwise comparison, the hypotheses are defined as follows.\n",
        "\n",
        "$H_0$: There is no statistically significant difference in the distribution of AUB between the two clusters being compared.\n",
        "\n",
        "$H_1$: There is a statistically significant difference in the distribution of AUB between the two clusters being compared."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 51,
      "id": "b5p7iq425vZF",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 174
        },
        "id": "b5p7iq425vZF",
        "outputId": "de939fbb-ee5f-453a-f40c-dd6e7d6500d1"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "     0       1       2    3\n",
            "0  1.0  0.0000  0.0000  0.0\n",
            "1  0.0  1.0000  0.0022  0.0\n",
            "2  0.0  0.0022  1.0000  0.0\n",
            "3  0.0  0.0000  0.0000  1.0\n"
          ]
        }
      ],
      "source": [
        "dunn = sp.posthoc_dunn(\n",
        "    scommerce_df,\n",
        "    val_col=\"AUB\",\n",
        "    group_col=\"Cluster\",\n",
        "    p_adjust=\"bonferroni\"\n",
        ")\n",
        "\n",
        "dunn = pd.DataFrame(dunn)\n",
        "\n",
        "print(dunn.round(4).to_string())"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "8RAheQOa58Xq",
      "metadata": {
        "id": "8RAheQOa58Xq"
      },
      "source": [
        "Dunn's post hoc test showed that all pairwise comparisons were statistically significant (*p* < .05). Therefore, the distribution of AUB differed significantly between every pair of identified clusters, indicating that each cluster represents a distinct pattern of Actual Usage Behavior."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "QhY8_A1WIcUm",
      "metadata": {
        "id": "QhY8_A1WIcUm"
      },
      "source": [
        "### [3] Insights and Conclusions"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "b3QIrxVU2VkY",
      "metadata": {
        "id": "b3QIrxVU2VkY"
      },
      "source": [
        "#### **Insights**\n",
        "\n",
        "After performing a clustering analysis, four distinct segments of Generation Z university students were identified based on their perceptions of social commerce: Highly Engaged Users, Regular Users with Positive Perceptions, Moderate or Occasional Users, and Low Engagement Users. The Elbow Method and Silhouette Analysis identified four clusters as the most suitable segmentation, indicating meaningful differences among respondents.\n",
        "\n",
        "The cluster profiles reveal a clear relationship between users' perceptions of social commerce and their actual usage behavior. Respondents who reported higher levels of Perceived Usefulness (PU), Perceived Ease of Use (PEU), Familiarity with Social Commerce (FSC), Social Presence (SP), Trust in Platform (TP), and Interaction Behavior (IB) also exhibited higher Actual Usage Behavior (AUB). Conversely, respondents with consistently low perception scores demonstrated the lowest platform usage.\n",
        "\n",
        "The largest cluster consisted of Regular Users with Positive Perceptions, suggesting that most respondents view social commerce favorably and engage with these platforms regularly, though not at the highest level. In contrast, Highly Engaged Users represented a smaller but strongly committed group with consistently high ratings across all behavioral constructs. Low Engagement Users formed the smallest cluster, indicating that only a limited portion of respondents maintain negative perceptions and low platform usage.\n",
        "\n",
        "Statistical inference further validated the clustering results. The Shapiro-Wilk test assessed normality, and since the assumption was violated, the Kruskal-Wallis H test was used instead of One-Way ANOVA. The test found a statistically significant difference in Actual Usage Behavior across the four clusters (H = 421.32, p < .001). Dunn's post hoc analysis showed that every pair of clusters differed significantly, confirming that each segment represents a distinct level of social commerce engagement.\n",
        "\n",
        "Overall, the findings suggest that positive perceptions of usefulness, ease of use, trust, familiarity, social presence, and interaction behavior are consistently associated with greater adoption and continued use of social commerce platforms among Generation Z university students."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "GqpH81vGVEZ-",
      "metadata": {
        "id": "GqpH81vGVEZ-"
      },
      "source": [
        "#### **Conclusions**\n",
        "\n",
        "This study successfully addressed the research question by identifying distinct behavioral segments among Generation Z university students through K-Means clustering. Four meaningful user profiles were discovered, each exhibiting different levels of perception toward social commerce and actual usage behavior.\n",
        "\n",
        "The results demonstrate that students with more positive perceptions consistently exhibit higher platform usage, while less favorable perceptions are associated with lower engagement. Statistical testing confirmed that Actual Usage Behavior differs significantly across all clusters, reinforcing the validity and practical relevance of the segmentation.\n",
        "\n",
        "These findings indicate that Generation Z users are not homogeneous. They differ in their attitudes, trust, familiarity, and engagement with social commerce platforms. Consequently, businesses and platform developers should adopt segment-specific strategies. Enhancing usability, trust, interactive features, and user familiarity may help encourage less engaged users to become more active while maintaining existing high-usage users.\n",
        "\n",
        "Although the study provides meaningful insights into social commerce behavior among Generation Z university students, its findings are limited to the surveyed population and variables included in the dataset. The original dataset was collected through voluntary online and offline recruitment, with online recruitment relying on Facebook and Zalo. This may introduce self-selection bias and overrepresent students who are more active on social media or familiar with social commerce. Consequently, the identified segments may not fully represent the broader Generation Z student population. Future studies may employ more representative sampling, expand the respondent population, incorporate additional factors, and explore alternative clustering techniques to further validate these segments."
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.14.5"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
