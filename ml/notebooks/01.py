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
        "This project investigates the social and behavioral factors associated with Actual Usage Behavior (AUB) in social commerce among Generation Z university students in Vietnam. Using survey responses from 757 participants, the study examines how perceived ease of use, perceived usefulness, familiarity with social commerce, social participation, trust in platforms, and purchase intention relate to actual engagement with social commerce platforms. The project aims to uncover meaningful patterns among users and provide insights into the factors associated with social commerce adoption and usage among young consumers.\n"
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
        "The first phase of the case study involves four sections – (1) dataset description, (2) data cleaning, (3) Exploratory Data Analysis, and (4) research question."
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
        "The dataset was built using an online survey administered by a research team from the Industrial University of Ho Chi Minh City and Thai Nguyen University of Economics & Business Administration. This survey took approximately 15-20 minutes to complete, and participation was completely voluntary. Respondents between 18 and 27 years old were chosen.\n",
        "\n",
        "Because the survey strictly filtered respondents to be of age 18 to 27, the results, analysis, insights, and conclusions cannot be generalized to that of the whole population, primarily because this demographic only includes their generation.\n",
        "\n",
        "The ZIP contains three files: CSV file containing all survey responses, docx file of the survey questionnaire, and an Excel file of the codebook containing variable definitions. Since the CSV file already contains all survey responses in a single table, no merging of multiple datasets is required for the analysis. The questionnaire and codebook serve as supporting documentation.\n",
        "\n",
        "The dataset is stored in a single CSV file. It contains survey responses collected from the chosen respondemts regarding their social commerce (S-commerce) usage behavior. Each row corresponds to one respondent, while each column corresponds to a demographic characteristic or a survey item measuring a specific construct related to social commerce adoption and usage (such as perceived usefulness, trust, familiarity, interaction behavior, and actual usage behavior).\n",
        "\n"
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
        "from scipy import stats\n",
        "\n",
        "import os\n",
        "import sys\n",
        "\n",
        "sys.path.append(os.path.abspath(\"..\"))\n",
        "\n",
        "# Import preprocessing wrapper functions\n",
        "from scripts.preprocessing import (\n",
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
        "outputId": "0e91af4a-67bf-4a88-939f-48461bdc2e4e"
      },
      "outputs": [],
      "source": [
        "# The dataset is first loaded into a dataframe using `pandas`.\n",
        "scommerce_df = load_dataset(\n",
        "    \"../datasets/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv\"\n",
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
        "The dataset contains 757 observations (rows), 31 variables (columns)."
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
        "outputId": "b3ded607-aa52-4217-9d96-31bc48088bb2"
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
        "The dataset contains demographic variables (Gender, Income, Area, Frequently) and survey construct variables measuring Perceived Usefulness (PU), Perceived Ease of Use (PEU), Familiarity with Social Commerce (FSC), Social Presence (SP), Trust in Platform (TP), Interaction Behavior (IB), and Actual Usage Behavior (AUB)."
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
        "outputId": "484e3298-bbff-41ea-86bf-9ca0ff52d57d"
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
        "The `Gender` variable contains 3 unique values.\n",
        "* 1 - Male\n",
        "* 2 - Female\n",
        "* 3 - Different"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "id": "jwM0uyQJe2Wa",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 210
        },
        "id": "jwM0uyQJe2Wa",
        "outputId": "a284f010-d1af-47bf-992a-bb2f27bf98e2"
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
        "The `Income` variable contains 5 unique values.\n",
        "* 1 - less than 100 USD\n",
        "* 2 - from 100 USD to less than 200 USD\n",
        "* 3 - from 200 USD to less than 300 USD\n",
        "* 4 - from 300 USD to less than 400 USD\n",
        "* 5 - above 400 USD"
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
        "outputId": "19dcfb4c-4ce5-4b14-e117-23a1a85ab34f"
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
        "The `Area` variable contains 3 unique values.\n",
        "* 1 - Urban\n",
        "* 2 - Suburban\n",
        "* 3 - Rural"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "id": "xyalc6gvgifq",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 210
        },
        "id": "xyalc6gvgifq",
        "outputId": "0913ef7c-c842-4b58-daad-ac37241904cf"
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
        "The `Frequently` variable contains 4 unique values.\n",
        "* 1 - Daily\n",
        "* 2 - Weekly\n",
        "* 3 - Monthly\n",
        "* 4 - Rarely Used"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "id": "KycxA7MogsBB",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 241
        },
        "id": "KycxA7MogsBB",
        "outputId": "b8d2686c-fd21-4c90-981b-efe320a314c0"
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
        "#### For the following survey variables, they use the following scale:\n",
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
        "* `PU2`: Using social commerce platforms increases productivity in finding products.\n",
        "* `PU3`: Using social commerce platforms enhances effectiveness when making online purchases.\n",
        "* `PU4`: Social commerce platforms are useful for online transactions."
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
        "outputId": "09ddb3b3-2ac7-4dfa-ba72-2ce5a1b4d2f9"
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
        "outputId": "1d21ffb8-271c-4988-b693-f386a246baa0"
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
          "height": 241
        },
        "id": "7VqxwWDO8NQV",
        "outputId": "678b7700-899f-48ab-bdf2-16e531befded"
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
        "* `FSC2`: Familiarity gained through reading news, blogs, and related materials.\n",
        "* `FSC3`: Familiarity gained through communication with others regarding social commerce.\n",
        "* `FSC4`: Familiarity gained through discussions, reviews, and shared experiences.\n"
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
        "outputId": "3191b29c-cb61-4cdc-a3cf-15565d884126"
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
        "* `SP2`: Frequency of socializing with friends through social commerce platforms.\n",
        "* `SP3`: Frequency of socializing with relatives through social commerce platforms.\n",
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
        "outputId": "6e9f891d-a3fd-450f-ea5d-adfd2e881361"
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
        "* `TP2`: Social commerce platforms act in users' best interests.\n",
        "* `TP3`: Social commerce platforms can be trusted consistently.\n"
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
        "outputId": "ba3559e4-f36b-4cec-d7bb-df531e2bb636"
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
        "These variables measure respondents' willingness to interact and share information with others through social commerce platforms.\n",
        "* `IB1`: Willingness to provide information to vendors.\n",
        "* `IB2`: Willingness to share shopping experiences and suggestions with friends.\n",
        "* `IB3`: Willingness to purchase products recommended by friends.\n",
        "* `IB4`: Willingness to consider friends' shopping experiences when making purchases."
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
        "outputId": "fdd56f1d-475d-44d6-dc93-2fbd37921c5b"
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
        "These variables measure respondents' actual usage of social commerce platforms.\n",
        "* `AUB1`: Using social commerce platforms is enjoyable.\n",
        "* `AUB2`: Uses social commerce platforms for safe online shopping.\n",
        "* `AUB3`: Spends significant time on social commerce platforms.\n",
        "* `AUB4`: Regularly uses social commerce platforms."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "id": "0d0cc677",
      "metadata": {},
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
              "      <th>AUB1</th>\n",
              "      <th>AUB2</th>\n",
              "      <th>AUB3</th>\n",
              "      <th>AUB4</th>\n",
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
              "      <td>4</td>\n",
              "      <td>2</td>\n",
              "      <td>4</td>\n",
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
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   AUB1  AUB2  AUB3  AUB4\n",
              "0     4     4     4     4\n",
              "1     3     3     3     3\n",
              "2     3     4     2     4\n",
              "3     5     5     5     5\n",
              "4     3     3     3     3"
            ]
          },
          "execution_count": 16,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "scommerce_df[['AUB1','AUB2','AUB3', 'AUB4']].head()"
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
        "Before conducting the analysis, the dataset was inspected to identify potential data quality issues that could affect the validity and reliability of the results. This process involved examining the variables for missing values, duplicate records, incorrect data types, inconsistent formatting, invalid values, and other anomalies that may interfere with subsequent analyses. Approriate cleaning and processing were applied as necessary."
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
        "Before any cleaning procedures were performed, the variables loaded from the dataset were examined against the questionnaire and accompanying study provided by the authors. This step was conducted to verify that all expected variables were successfully loaded and that each variable was properly documented and corresponded to the constructs described in the study to ensure consistency."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "id": "PlIAL3dltRnO",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PlIAL3dltRnO",
        "outputId": "de7378ad-66f8-4e09-e697-92a3914950ae"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Index(['Gender', 'Job', 'Income', 'Area', 'Frequently', 'PU1', 'PU2', 'PU3',\n",
              "       'PU4', 'PEU1', 'PEU2', 'PEU3', 'PEU4', 'FSC1', 'FSC2', 'FSC3', 'SP1',\n",
              "       'SP2', 'SP3', 'SP4', 'TP1', 'TP2', 'TP3', 'IB1', 'IB2', 'IB3', 'IB4',\n",
              "       'AUB1', 'AUB2', 'AUB3', 'AUB4'],\n",
              "      dtype='object')"
            ]
          },
          "execution_count": 17,
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
        "As discussed in the Dataset Description section, the variable `PEU4` is not documented in the codebook, questionnaire, or the associated research paper. Since its meaning and purpose cannot be verified, the variable is removed from the dataset to avoid introducing ambiguity into the analysis."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "id": "RNVxTX89IgrA",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 253
        },
        "id": "RNVxTX89IgrA",
        "outputId": "0989fa91-8352-4793-ab33-1ddeb65bb15c"
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
          "execution_count": 18,
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
        "The code above drops the `PEU4` column from the DataFrame and then displays the dataset's dimensions and a preview of the first few rows to confirm that the column has been successfully removed."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "id": "RW-JyjohT9U-",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RW-JyjohT9U-",
        "outputId": "f4c66856-2617-41aa-e332-927667718429"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Index(['Gender', 'Job', 'Income', 'Area', 'Frequently', 'PU1', 'PU2', 'PU3',\n",
              "       'PU4', 'PEU1', 'PEU2', 'PEU3', 'FSC1', 'FSC2', 'FSC3', 'SP1', 'SP2',\n",
              "       'SP3', 'SP4', 'TP1', 'TP2', 'TP3', 'IB1', 'IB2', 'IB3', 'IB4', 'AUB1',\n",
              "       'AUB2', 'AUB3', 'AUB4'],\n",
              "      dtype='object')"
            ]
          },
          "execution_count": 19,
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
        "Based on the dataset columns displayed above, all expected variables from the questionnaire and supporting documentation were successfully loaded from the CSV file. However, an additional variable named Job was identified. This variable is not documented in the questionnaire, codebook, or accompanying research paper, nor is it utilized in the study. To determine whether the variable contains any meaningful information relevant to the analysis, its contents will be further inspected."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "id": "8e9657bc",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8e9657bc",
        "outputId": "33f7532d-3bed-483d-c465-638bd1c34606"
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
      "id": "KljZOxlOvVKR",
      "metadata": {
        "id": "KljZOxlOvVKR"
      },
      "source": [
        "Further inspection of the Job variable revealed that all 757 observations contained the same value. Since the variable exhibits no variation and is not documented in the questionnaire or research paper, it provides no meaningful information for analysis. Therefore, the variable was removed from the dataset."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "id": "c6d04760",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 253
        },
        "id": "c6d04760",
        "outputId": "ba1aba9c-4f71-4ef7-f894-1310124487d3"
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
          "execution_count": 21,
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
        "As a result, the dataset now has 30 variables/columns, all which are properly documented."
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
        "In this part, missing values or null entries were examined across all variables in the dataset given they might affect the whole analysis if not handled properly."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "id": "QG1-OpP5yR0N",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 994
        },
        "collapsed": true,
        "id": "QG1-OpP5yR0N",
        "outputId": "46ce09f3-7b22-4ea0-e0e4-511a4eb0e935"
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
          "execution_count": 22,
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
        "As shown above, no missing values were identified across any of the variables in the dataset. Therefore, no additional cleaning, imputation, or removal of observations was required, and all records were retained for further analysis. This finding is consistent with the dataset documentation, which states that the authors had already performed a data screening process prior to publication. During this process, incomplete responses and low-quality submissions showing straight-lining behavior were removed, resulting in a finalized dataset containing only valid responses."
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
        "The dataset was examined for duplicate records to identify observations with identical values across all variables. Detecting duplicate records helps determine whether any entries may have been unintentionally repeated in the dataset and whether further inspection is necessary before analysis."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "id": "53XmtQFR0t1y",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        },
        "id": "53XmtQFR0t1y",
        "outputId": "09c3480e-9c0d-4c66-82f3-56d2ad54b732"
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
          "execution_count": 23,
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
        "As shown above, several observations were identified as duplicates based on the available variables in the dataset. However, the dataset does not contain a unique identifier that can distinguish individual respondents. As a result, it is possible for different participants to share identical demographic characteristics and survey responses, causing their records to appear as duplicates. Given that the dataset documentation states that a data screening and cleaning process had already been conducted by the authors prior to publication, these duplicate records were retained and assumed to represent distinct respondents rather than duplicate submissions."
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
        "The data types of each variable were examined to ensure that they were stored in a format appropriate for their intended use in the analysis. Incorrect data types may lead to errors during data processing and statistical analysis, potentially resulting in inaccurate or misleading results."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "id": "vFsdJMt37LFr",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 994
        },
        "id": "vFsdJMt37LFr",
        "outputId": "0f4ff01e-1aca-4687-e097-3a2263eadfa5"
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
          "execution_count": 24,
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
        "Given the results above, all variables were loaded using the expected numerical data types. This includes the demographic variables (Gender, Income, Area, and Frequently), which were numerically encoded by the dataset authors despite representing categorical or ordinal characteristics. Therefore, no data type conversion was necessary."
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
        "Although the data type verification confirmed that all variables were stored using the expected numerical data types, it is still necessary to verify that the recorded values themselves are valid. In datasets containing categorical variables stored as text, inconsistencies may arise from different representations of the same category (e.g., \"Male\", \"male\", or \"M\"). However, since all variables in this dataset were numerically encoded by the authors, the focus of this verification is on examining the unique values present in each variable.\n",
        "\n",
        "For the demographic variables, the observed values should correspond only to the categories defined in the dataset documentation. Similarly, the survey response variables are based on a five-point Likert scale and are therefore expected to contain only the values 1, 2, 3, 4, and 5. This verification step helps ensure that no unexpected, invalid, or improperly encoded values are present in the dataset, which could otherwise affect subsequent analyses. This also serves as a range check, as values outside the expected categories and response scales can be identified through this process."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "id": "WK_bcRn6-MZK",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 959
        },
        "id": "WK_bcRn6-MZK",
        "outputId": "ad362148-33f5-46d7-9f5e-02a8263ea46a"
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
          "execution_count": 25,
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
        "The values shown above indicate that all variables are within their expected format and range. The demographic variables contain only their documented numerical representations, while the survey response variables contain only the values 1, 2, 3, 4, and 5, consistent with the five-point Likert scale used in the questionnaire. Therefore, no invalid or unexpected values were identified, and no observations were removed as part of this verification process."
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
        "The questionnaire measures several constructs using multiple survey items. Rather than analyzing each item individually, composite scores were created for each construct by averaging the responses of their corresponding items. This approach is commonly used in survey-based research involving Likert-scale measurements, as it provides a single representative score for each construct while preserving the overall response pattern of the respondents. This is also in line with how the authors analyzed the constructs in the research paper."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 26,
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
        "Firstly, `scommerce_df[['PU1', 'PU2', 'PU3', 'PU4']]` is used to select all questionnaire items that measure the Perceived Usefulness (PU) construct. The average of the selected items for each respondent is then calculated using `.mean(axis=1)`. Setting axis=1 performs the calculation row-wise, producing one composite score per respondent. And lastly, `scommerce_df['PU'] = ...` is used to create a new column named PU and stores the computed composite score. The same approach is applied to the other remaining constructs (PEU, FSC, SP, TP, IB, and AUB)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 27,
      "id": "T6js7Yy5HN6N",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "T6js7Yy5HN6N",
        "outputId": "467d7c80-9bf7-4158-a0aa-233fdac925ca"
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
          "execution_count": 27,
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
        "After creating the composite scores, `scommerce_df[['PU', 'PEU', 'FSC', 'SP', 'TP', 'IB', 'AUB']].head()` displays the first five rows of the newly created variables to verify that the aggregation was performed correctly."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "RMQuiOvW4e-V",
      "metadata": {
        "id": "RMQuiOvW4e-V"
      },
      "source": [
        "The questionnaire measures each construct using multiple survey items. Following the methodology described in the original research paper, these items were aggregated by calculating their mean to create a single composite score for each construct. This approach assumes that the items within a construct capture the same underlying concept and can therefore be combined into a representative measure. Aggregating the items reduces the number of variables in the dataset, simplifies subsequent analyses, and improves the interpretability of the results while preserving respondents' overall response patterns. The resulting composite variables (`PU`, `PEU`, `FSC`, `SP`, `TP`, `IB`, and `AUB`) are used in the remaining stages of the analysis."
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
        "Before analyzing the factors influencing social commerce behavior, it is important to first understand the demographic composition of the dataset. The demographic variables: `Gender`, `Income`, `Area`, and `Frequently` (Frequency of Social Media Usage), were analyzed individually to identify patterns within the sample and determine whether any demographic groups are disproportionately represented. This provides important context for interpreting the results of the subsequent analyses and assessing the representativeness of the dataset."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "4dkLKxEmAWT2",
      "metadata": {
        "id": "4dkLKxEmAWT2"
      },
      "source": [
        "#### Gender Distribution\n",
        "\n",
        "The distribution of respondents by gender was examined to determine whether the sample is balanced across gender groups."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "id": "qg-2NIQfBPkm",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 175
        },
        "id": "qg-2NIQfBPkm",
        "outputId": "6351b0b7-710a-4873-d17a-4820a0aaa5ec"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjMAAAHACAYAAABXvOnoAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjksIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvJkbTWQAAAAlwSFlzAAAPYQAAD2EBqD+naQAAPYJJREFUeJzt3Qm8TPX/x/EPyr7UXcgWIbsQ0qZUVKSI9h/l14Is/UpR0oYkQmWrlFYliUolJflVspU1tMgu2738RNnC/T/e39//zG/u6uK6c7/m9Xw85jEz58ycmTnnzDnv813OyZWUlJRkAAAAnsod6S8AAABwLAgzAADAa4QZAADgNcIMAADwGmEGAAB4jTADAAC8RpgBAABeI8wAAACvEWYQVXLCOSJzwndAzsS6ARwdwgxyjHbt2lmVKlVCt6pVq1rdunWtdevW9uabb9qBAweSvf7SSy+1hx56KNPTnz59uj344IOHfZ2mqWkf7eekZ+fOndazZ0/74Ycfkv1m3XIKzWP9Vs33s88+2+bMmZPqNXPnzk22nIJbzZo17aKLLnK/MSEhwU5Ex2t5pbVuHI2sWlfTs2PHDhs1apRde+211qBBA6tdu7ZdddVV9uyzz7px2Smn/XcQWSdF+POBZKpXr26PP/64e3zw4EH7448/7JtvvrEBAwa4Df1zzz1nuXP/N4OPGDHCChcunOk5+Prrr2fqdZ07d7Zbb701y5fMTz/9ZB999JG1adMmNCz4rTnFt99+ax988IGbB+eff75bHul57LHHrEaNGqHnf/31l82fP99Gjx5tq1evtgkTJmTTt/ZfWutGTvPrr79ax44d7e+//7a2bdtarVq1LE+ePLZo0SJ74403bMqUKfbuu+9abGxspL8qohBhBjmKwkmdOnVSHW1WqFDB+vfvb5988oldc801bnhGO9pjcfrpp1t2qVSpkuUkwdG1SsPKli172O+eclldcMEFtn//fnv55Zftt99+y3G/D0dn3759du+997rwMnHiRIuJiQmNO/fcc13pTMuWLW3YsGHWp08fZjOyHdVM8IKOBEuUKOGO/NIrUg+CzllnneU2sA888IBt2bLFjVNx9Lx589xNVSKqKgmqSzTNSy65xFWrfPfdd6mqmURHo08++aQrWq9fv76rrtq+fXuGRd7B9IPPCkp7dB+8NuX7tNMYOXKkXXnlle7I9/LLL3clHYcOHUr2Wb1793bDGzdu7F5300032ZIlSzKchyrpevvtt+3qq69280jvHTx4sPtM0e8O5meTJk2Ougi/aNGi7j5Xrlypjuo1j3Xr0qWLrV+/Ptn7dHQf/O5GjRrZE088YX/++WdovObl2LFj3bxXNZhKjhRwg+8fUAmBwpheo3ClEiSV8AWGDx9uTZs2tX//+99uXqh67IorrrAPP/ww2XQ2btxoXbt2tXr16rnpvPbaa2n+XpVAaWeu6Wieavqa1wHN0/bt27sQoM/R67TjV4mjpLdurFu3zjp16mQNGzZ01Tk33nijff3114ed/xmtq/rNmo8zZ85M9h6Vemq4StbS8tlnn9nKlSvt0UcfTRZkAgq+d999d6pxX375pVsWWqaah/peu3fvjuiyuO2221yJqNbD5s2bJxsPf1EyAy+oaum8886zTz/91LXrOOmk5KuuNsJqc6DqEW3EN2/ebM8884zdf//9bgeojVePHj3ca/VYJQbLli0LVVc98sgjtnfvXrcD/Pjjj9PcmGuH8vTTT7sdg0KASh7ee+89d7R6OKqO0U61b9++7l47qLQaf2rnpWJ7bbjVZkg7OlWtacffr1+/0Gs///xzq1ixovveet/AgQOtW7du9tVXX6X7ffS5qsq466673E5u+fLlLjipiuOVV15x8+60006zF154wc2TM844I8PfpIAV3o5JwUNhccyYMS4sBe9XlZPClkrX9D31Hn3GzTff7L6PqiUURLW8tOPVTnXVqlXutXv27HH3geeff94tB80T7Vx1r/Y5uhe151DpwC233GL33Xefm296j+apllX+/Pnd6/QeLQvtgEuXLu2+sz5bO13NV+1wFaC1nmm+a/3TdBUwtI4EXnrpJddeRK/t1auXm5fagW7atMmeeuqp0OuWLl1qW7dutXvuuceVPuo7aXkp0KS1bmjeKvwVL17cBg0a5L6H2o3p+2pdLFeuXLrLJaN1VSFR09R8v/DCC0PvUXgoX768CwtpUSgpVqyYe396tF6F0/9IBxQKKSrV+f3339280ndRGAnCbnYvCwW3fPnyuXVf087M/xc5H2EG3oiLi3NHnaoK0eOUYUY7qg4dOljevHndsFNOOcV+/PFHt7NXeAna16SsGtGOTyUCGTn11FPdRrZgwYKh5ypd0M5IpTqHo88Oqlx0n1b1i6Y1a9Ys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uXIVlhFwgopYm5n58+db4cKF7ZxzzgkNU2mMPProo1avXj3L9f99gHV/9tln26JFi6x169Yu8Nx1112h95UsWdJKlSrlhpctWzYCvwZAVlOV87p1a23u3Nn25puv2aFDB+2SS5rYnXd2sssuu9y+++4b69z5TsuTJ4/lzp3bBg161ooWLere27nzv1xbm6uuusw9V+nOwIFDWUjACSpiYUalKKVLl7YPP/zQXnzxRfv7779dULn77rstISHBKlWqlOz1sbGxtmLFCvd469atVrx48VTjN2/enOFncn6UYxPMP+YjssOWLZtdG5i8efNav34DbNOmjfbcc4Nt//599o9/3Gbbt2+z7t17Ws2atWzKlI/sqaf62muvjbVTT42x339fbyVKnGaPPPKEHThwwIYOHWTDhz9rDz30CAsPxx3byigKM2r/snbtWnv33XdtwIABLsA89thjrrfBnj173AYsnJ6rwbAEG7j0xqd3Nkad8A3HLjb28GdjBI5VXFwRmzt3rqsyCkppCxfOZz169LB9+3Zb9erVrGPHO9zw886r76qeZ8z43G655RYbOPBJe/31110HASlRIsbatm1rPXven+pACDhe2FZGQZg56aST7M8//3QNf1VCIxs3brRx48ZZuXLlUgUTPc+fP797rPY0aY3PqNulTsFPicKx0fzTn3Pbtl32/53OgOMsj23b9mfoWUzMaa7b9eLFS+z662+yxMRdofVSVUmrVq21BQt+dAdLcXGl3XgpUeJ017bm559XWu7cdM/G8cW2MusPbHJsmImPj3ehJAgycsYZZ9imTZtcO5rExMRkr9fz4IiqRIkSaY7XNDPCDjhraD4yL3G8qa2Mul9PmvRp6EDm119/dSU1cXHFbc2a1cnWQ7Wvufzy6hYb+9/twOrVq61Klf+ed2bNmjXu/rTTSrPuItuwrcw+Eat3UfGvjrC0wQmsWrXKhRuNW7hwYeicM7pfsGBBqMhY92pAHFAA0i0YD8B/tWqd5Q54nn66n61bt8Z1t9aJ8G655Va75ppWNnnyhzZ16qe2YcN6Gzx4sG3evMl11S5evIQ1bHi+DRrU337++Sf7+efl7rEaDZ966qmR/lkAjoNcSUFiiACdO+aPP/6wJ554wrWZUVdsNQBWQ+CmTZvaVVddZTfddJNrV6Nu2V988YUVLFjQBZ127drZ448/brVq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",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "gender_counts = scommerce_df['Gender'].value_counts().sort_index()\n",
        "\n",
        "gender_labels = {\n",
        "    1: 'Male',\n",
        "    2: 'Female',\n",
        "    3: 'Different'\n",
        "}\n",
        "\n",
        "gender_counts.index = gender_counts.index.map(gender_labels)\n",
        "\n",
        "ax = gender_counts.plot(kind='bar')\n",
        "\n",
        "for container in ax.containers:\n",
        "    ax.bar_label(container)\n",
        "\n",
        "plt.title('Distribution of Respondents by Gender')\n",
        "plt.xlabel('Gender')\n",
        "plt.ylabel('Count')\n",
        "plt.xticks(rotation=0)\n",
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
        "The results show that **Females** make up the largest proportion of the sample, with **588 respondents**, followed by **Males** with **167 respondents**. The **Different** group represents the smallest proportion of respondents, with **2 respondents**. This distribution provides an overview of the demographic composition of the study participants and may be considered when interpreting subsequent analyses."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "TuWCoMt4FLdz",
      "metadata": {
        "id": "TuWCoMt4FLdz"
      },
      "source": [
        "### Income Distribution\n",
        "\n",
        "Monthly income categories were analyzed to understand the economic background of the respondents."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 29,
      "id": "bzyfTWSwCent",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 465
        },
        "id": "bzyfTWSwCent",
        "outputId": "00f5c839-04ac-4fdd-fc17-32d98243ca45"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjMAAAHBCAYAAACc4DpNAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjksIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvJkbTWQAAAAlwSFlzAAAPYQAAD2EBqD+naQAAR8JJREFUeJzt3QucTfX+//HPRO4kM4hIRXK/Xzoq0V1xCHFU1FFRUedUIuRaKEKRinRXlGvofrrJUcg1SRG5JMyQ+539f7y//7P2b8+YGeM2e63Zr+fjMY+Zvdbae6+9vmvWeu/v+n6/Ky4UCoUMAAAgoM6K9goAAACcCsIMAAAINMIMAAAINMIMAAAINMIMAAAINMIMAAAINMIMAAAINMIMAAAINMIMYo4fxon0wzrAn9g3gBNHmIGvtG3b1i699NLwT7ly5ax69erWvHlze+utt+zw4cPJlr/66qvt8ccfz/Drf/HFF9atW7fjLqfX1Guf7PukZefOnda1a1f74Ycfkn1m/fiFtrE+q7Z7jRo17Pvvvz9mmblz5yYrJ++nUqVKVr9+ffcZExMTLSs6U+WV2r5xMk7XvppWmes34DfZo70CQEoVKlSwPn36uL+PHDliO3bssFmzZtmgQYPcgf65556zs876/zn8hRdesHz58mV4I77xxhsZWu6BBx6wdu3anfbC+fnnn+2DDz6wFi1ahKd5n9Uvvv32W5s6darbBvXq1XPlkZbevXtbxYoVw4/37NljCxYssDFjxtiaNWts4sSJmbTWwZfavgEgYwgz8B2Fk2rVqh3zbfPiiy+2AQMG2MyZM+3vf/+7m57eifZUXHDBBZZZypQpY36yfft291u1YSVLljzuuqcsq8svv9wOHjxor7zyiq1atcp3nw9A1sNlJgTGHXfcYUWLFrUJEyakWaXuBZ0qVarYZZddZl26dLHNmze7ebo0MG/ePPfjVZd7Ved6zYYNG7rLKv/973+Pucwkhw4dsqeeespq165ttWrVcpertm3blu7lh8iqef14tT367S2b8nkHDhywUaNG2Y033miVK1e266+/3tV0HD16NNl79ezZ001v0KCBW+4f//iHLV26NN1tqJqud955x5o0aeK2kZ777LPPuvcUfW5ve1577bUnfTmlQIEC7ndcXFx42q+//modO3Z021g/nTp1svXr1yd73ptvvhn+3FdeeaX17dvXdu/eHZ6vbTlu3Di37XUZTDVHCrje+ns++ugjF8a0jMKVapBUw+cZOXKkXXfddfb111+7baHLYzfccINNmzYt2ets3LjROnfubDVr1nSv8/rrr6f6eVUDdfPNN7vX0TbV62tbe7RN77rrLps8ebJ7Hy3XtGlTV+Moae0b69ats/vuu8/q1q1rVatWtdatW9s333xz3O2f3r6qz6ztOHv27GTPUa2npqtmLSM2bNjglv/444/toYcectu6Tp069sQTT9jevXuTtQFSjWijRo3cPqft/uqrryZrG6T/udtuu81tZ33WRx991P7888/w/ClTprh9Quuomiv9re345Zdf2urVq+3OO+9020ev/eGHHx5Tho888ohbNy2jZZcvX56hz4jgIMwgMHRp6W9/+5s7YadsOyM6CKvNgU7+qhXo3r27a++hA6N3OUc1Ofp57733kl0e0eUqHfB10tNBOTU6aP/000/29NNPu2V1Urj33nuTnbTSo/fT64t+p3Z5SQd4nbzGjh1rt956q7388svu5K5LaymX//TTT10bIJ08hg0bZklJSfbggw+muz56X12uU1B56aWX7Pbbb3fhQJeU9N76ff/994e3yfEugSlgqSy8H9XqfPbZZ+5kpRPXRRdd5JbTJSeFra1bt9ozzzzjAoiCTJs2bdw0L4gOGTLErZOer7Cjyy5PPvlksvd8/vnn3XO0Te655x5XlpHtoF588UV38lKN0YgRI9zraFspIOzfvz+8nNr09O/f34UHhcISJUq41/ntt9/cfJ2QFaAVwrQOvXr1cqFl0aJFydZn9OjRbp72TZWX1l/7n6ZFWrZsmftcOvErrGbLls2Vl0JWavuGtq3C3759+2zw4MHucxUsWNCVz9q1a9Mtl/T2VYXEIkWKuG0bSUHuwgsvdIHiRGhdzz//fLd+d999t02aNMntWx6tu3705UDbp2XLli5Aa5t779u+fXsrVqyY24/1f6ttrODm7Rui/Uv/y9qP9Pq5c+d2X1b0/6IAqdfW59Ln3bRpk3uOApyW17ZQeQwdOtRtV5WRV87IIkKAj9xxxx3uJy2DBw8OlS1bNpSYmOgeN2zYMNStWzf39+jRo0PVq1cPHThwILz8119/HRo5cmTo6NGjqb7+999/715v1KhRyd5Hr6nX9ujvevXqhfbs2ROe9vnnn7vnfvnll2mu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YWbs2LHu9gMarC7lpSbdfoAwAwAAfB1mXnvtNXvsscfs7rvvPv1rBAAAcKa7Zh84cMCuv/76k3kqAABA9MNMkyZN7N1333VdtAEAAAJ3mWn37t02adIkmzlzppUoUeKY8Wbeeuut07V+AAAApz/MXHjhhXbfffedzFMBAACiH2a88WYAAAACGWa6d++e7vxBgwad7PoAAABk/l2zDx8+bGvWrLGPPvrIChUqdDpeEgAA4MzVzKRV86LB9H799deTeUkAAIDo1cx4brzxRvv8889P50sCAABkTpjZu3evvf/++3buueeerpcEAAA4M5eZypUr5+7BlJLuov3UU0+dzEsCAABkXphJOSiego0GzitTpozly5fv5NYEAAAgs8JMnTp13O/ff//dfvvtNzt69KhddNFFBBkAABCMMLNz50431swXX3xh55xzjh05csT27NljtWvXtlGjRln+/PlP/5oCAACcrgbAahezadMmN67M3Llz7YcffrAZM2a4RsAMmAcAAHwfZr788kvr27evXXzxxeFpai/Tu3dvV1sDAADg6zCjXktnnXXsU9UQWJecAAAAfB1mrr76auvXr5+tW7cuPE2NgXX56aqrrjqd6wcAAHD6GwA/9thj1qlTJ7vhhhusQIECbtqOHTusfv361qtXr5N5SQAAgMwJM2vXrrXixYvb22+/bb/88ovrmq3LThdeeKGVLl365NYCAADgTF9mCoVC7jJSo0aNbNGiRW7apZdeajfddJNNnjzZGjdubE8//bRbDgAAwHdhRqP+qiu2xpHxBs3zvPjii2761KlTbfz48WdiPQEAAE4tzOgmkmoP07BhwzQbBXfp0oUwAwAA/Blm/vjjD6tSpUq6y1x22WW2fv3607FeAAAApzfMxMfHu0CTHo0KXLBgwYy+JAAAQOaFmeuuu85Gjhxphw4dSnX+4cOH7YUXXrArrrji1NcKAADgdHfNfuCBB6xly5bWvHlza9u2rVWqVMndUFLjy/z00082btw4d7PJwYMHZ/QlAQAAMi/MaHA8NQJ+9tlnXRfsffv2uenqiq1Qoy7aDz74oCUkJJz6WgEAAJyJQfPUHkZjzeiGkmrou3PnTjftggsusGzZsp3ISwEAAETvdgY5cuRgtF8AABDcG00CAAD4RVTDjO7zdPfdd1v16tWtQYMGNnbs2PA8Xca66667rFq1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hQsXuumar+VKlCjhekYBp9Pzzz9r8+fPtWHDRlqfPk/ZjBlT7YMPprj9rm/fJ+yOO+6yN94Yb5dcUta6dv1XsgAOAIjhMJMtWzZ78cUXXa8lDYw3ffp0GzVqlBUvXtzNV3AZOXKkG3dGAUftazQ/jkYxOI127txhM2d+YN269bQKFSpZrVp1rHXrO2z58mX23//+1y666GJr1KixnX9+Cbvvvs62detW+/331ZQBAGQy37SZ+eWXX5I9LlWqlI0bNy7N5a+66ir3A5wpS5cutnz58ln16jXD09q2vcs1rJs9+wtbs2a1W6ZSpSr24YczLG/evFa8eAkKBABiNcwAfrNx4x923nnF7eOPZ9rbb79uhw4dtptvbmJ33tne3S/s448/tQceuMfVJKpWcMiQ59LtUQcAODMIM0Aa9u7daxs2rLPp06dYjx59bOvWJBsyZKAbIqBVq+a2bdtWe/jhrlaxYmWbNm2SDRzY3157bZyde24htikAZCLCDJCGbNmy2549e6xPnwF23nnF3LTNmzfZ1KmTbN261XbxxWWsRYtWbnrXrj1dz6YPP5zuGgUDAGK8ATDgB7pre44cOcNBRkqWLGWbN292QweUKXNJePpZZ51lZcqUPe4tNQAApx9hBkhDxYqV7ODBA7Zu3drwtLVr17hhAzSA4++/r0m2vJbzetwBADIPl5mANFxwwYVWr94VNnBgP3v00cddG5lx4960u+6620qVOt+6dXvcypWr4HozzZgxzTZv/tN11QYAZC7CDJCO3r2fsuHDB7teS2r4qzYyLVu2tsKFC9jmzVtdL6ctW7a4QfOef/5lGv8CQBQQZoB0aJyZXr36J5vmjc3YpEkza9y4GdsPAKKMNjMAACDQCDMAACDQuMyELKn20FkWdPMfrR/tVQCAQKBmBgAABBphBgAABBphBgAABBphBgAABBphBgAABBphBgAABBphBgAABBphBgAABBphBgAABBphBgAABBphBgAABBphBgAABBphBgAABBphBgAABBphBgAABBphBkCgHDx40Nq2bWULF/4QnrZ48WLr2LG9XXfdldamTXObMWNaVNcRQOYizAAIjAMHDljfvj1tzZrV4WlbtybZvffea9Wr17TXXnvH7r67ow0fPsTmzJkd1XUFkHmyZ+J7AcBJU4Dp1+8JC4VCyabPmvW1JSQk2H33dTLN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z09efvnlUPPmzV1A0D56//33h3777TfflceRI0dC119/vTvgr1mzJvT111+HatSoEZo+fXqWKY933nnHfZHRuulH+58+p9/KItLMmTPd/3HKMBP0sjgRP/zwQ6hLly4uWGRERs4Z4h1zdKyMDDOPPfZYsu2tL8qXXnppaN26de6xgkzk8tOmTXP7g18RZjJpJ73vvvtC5cuXd98+vJ0lZc2KdsxZs2aFd8yUevbsGRo/fryblzLM/POf/3SBJpLCjU4yhw8fDlWuXNkld8+hQ4fc+ixcuDAURDo4Dxs2zB3EUgszH374YahixYqhSZMmhVauXBl64oknQrVq1QolJSWd0DKnI8Bq/RITE8PTZsyY4b7lyJ49e1zZRJb3qFGjXNmlN89vVNOhA58Cmw7G2s8bNGjgPoOfymPz5s2u1nLXrl3haZ06dQr16dMny5THF1984f4/FNZWr17t/k+0XVU2fioLz19//RWqX79+qEWLFslOrlmhLE5039S+qOOy/p8ij9epycg548CBA6Ebb7wx9O2337ogEhlOFOonTpyYbHkto+OTapP12pHnqvXr17tpWk8/IsycIfpGrhNs69at3T+dDgY6saVHYUW1JbqMpBCiasTUpBZm9E3s3XffTTZNB+2+ffuGtm3b5pZP+f5/+9vfQh999FEo6FILM/omGXkZTd/IFSBGjx59Qsucqh07driDTSQdLFSDIQsWLHAHLx10IstX+0x687SufrZ161ZXLqpB81N5pPwf1ReNOnXquJN3Vi4PXeJ8//33fVkWjz/+eGjo0KEuyESGmaxaFsezdu1aV+ukY8Qtt9zi9k19IT2Zc8bzzz8feuSRR9zfKcOMXj/lVQKVuy67/vjjj+7/N/JyuGqANO1M1tCdikA3APaz+fPnuwZV559/vn399df25JNPWunSpdN9zlNPPeUaXKmn1ogRI+yGG26wJUuWnNKIyBpIcP/+/eHHqc3PavSZfvrpJ6tXr154mrrr6/GiRYsyvMzpoDGP1PDOowZ048aNc43qjndbDvVsS++WHX6mBodyzjnn+Ko8Il199dV22223uQbx+l/LiuVx5MgR+/DDD13PSH1Ov5WFGq7+8MMP9sADDxwzL6uVRUZdcMEF1qtXL3feUAeXhx9+2L766qsTPmesWrXKJkyY4DrKpEbnhRM5Z6Q3wr4fEGbOkDJl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LrvsMitfvrzVq1fP1dTo7+uvv96FKHn11VetVatW1qRJEytVqpSrvalfv74LWACCI3u0VwBAbIqPj7d8+fKFH+vvw4cPu78VNipWrBiel5CQ4GpQkpKSbPv27S6oeM4++2yrVKmS/fbbb+FpJUuWDP+dK1cuV3sT+fjgwYPubz1Hl6QUnjyHDh2yK6644ox8ZgBnBmEGQFTkyJHjmGlem5ns2VM/NOXMmTPNWh61pfGopibSWWedlebzdFmpWbNmyaYr8AAIDi4zAfAdXfJRexbPX3/95S4Z7dixw9XSLF68OFlNyk8//WQXXXTRCb+PnrNhwwb3ft6PamlmzZp12j4LgDOPmhkAvqMeSurdVLZsWdcuZvjw4VaiRAn3o/YwauBbpEgRFz7UmPfAgQOufc2J0mvdfvvtVrlyZWvQoIF9+eWX9sYbb9ibb755Rj4XgDODMAPAd9QgWN2n+/XrZ7t377Y6deq4ACPt27d309QDSr/Vdfvtt9+2QoUKnfD7VKtWzQYPHuxGJdbvCy64wIYOHWq1a9c+A58KwJkSF4oc2AEAACBgaDMDAAACjTADAAACjTADAAACjTADAAACjTADAAACjTADAAACjTADAAACjTADAAACjTADAAACjTADAAACjTADAAACjTADAAAsyP4f1qnkVto9OHEAAAAASUVORK5CYII=",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "income_counts = scommerce_df['Income'].value_counts().sort_index()\n",
        "\n",
        "income_labels = {\n",
        "    1: '< $100',\n",
        "    2: '$100 - $200',\n",
        "    3: '$200 - $300',\n",
        "    4: '$300 - $400',\n",
        "    5: '> $400'\n",
        "}\n",
        "\n",
        "income_counts.index = income_counts.index.map(income_labels)\n",
        "\n",
        "ax = income_counts.plot(kind='bar')\n",
        "\n",
        "for container in ax.containers:\n",
        "    ax.bar_label(container)\n",
        "\n",
        "plt.title('Distribution of Respondents by Income')\n",
        "plt.xlabel('Income')\n",
        "plt.ylabel('Count')\n",
        "plt.xticks(rotation=0)\n",
        "\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "Bq66061Ttb1f",
      "metadata": {
        "id": "Bq66061Ttb1f"
      },
      "source": [
        "The bar chart shows that the majority of respondents belong to the **< \\$100** income category, comprising **672 respondents**. This is followed by the **\\$100-\\$200** income group with **68 respondents** and the **\\$200-\\$300** income group with **12 respondents**. Only a small number of respondents fall within the higher income categories, with **2 respondents** in the **\\$300-\\$400** range and **3 respondents** earning **more than \\$400**. Overall, the sample is heavily concentrated in the lowest income bracket, indicating that most respondents have relatively low income levels. This distribution is expected since the study's participants are university students who may have either limited personal income or rely on financial support from their families."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "xQJ5XyCzCyeW",
      "metadata": {
        "id": "xQJ5XyCzCyeW"
      },
      "source": [
        "#### Residential Area Distribution\n",
        "\n",
        "The residential location of respondents was examined to determine whether the sample is concentrated in urban, suburban, or rural areas."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "id": "dWRKU95tDOzO",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 465
        },
        "id": "dWRKU95tDOzO",
        "outputId": "d70f769f-eebb-4283-eecd-d4c0764487eb"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjMAAAHACAYAAABXvOnoAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjksIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvJkbTWQAAAAlwSFlzAAAPYQAAD2EBqD+naQAAO1JJREFUeJzt3QeUFFX+9vHfOEpGhZkBSYqCgaCAJFHERBBEUcCEC8uKChKMIEFFTCAKihJUjAjsqoAJzKKrIqgIAgIGggiCyAyIBIky73nu/63enp4IDNN9p7+fc/rMdFV3dXXV7aqn7r1VlZCenp5uAAAAnjos2jMAAABwMAgzAADAa4QZAADgNcIMAADwGmEGAAB4jTADAAC8RpgBAABeI8wAAACvEWZQYGLh+oyxMA+ITZSN2ML6wP4gzMDp3LmznXzyyaHHKaecYvXq1bP27dvbSy+9ZHv37s2wpM4//3wbMGBAnpfezJkzrX///rm+TtPUtA/0c7KzZcsWu+OOO+ybb77J8J31iBVaxvquWu6nn366ffnll5le89VXX2VYT8Gjdu3a1qxZM/cdU1NTrTA6VOsrq7JxIPKrrEYaPXp0luu8bt261qpVK3vssccy/T4PVlDO9Dcneo3mL78tW7bMrr766oP6rF9//dW957XXXsv1ta+++qp7bY8ePQ5ofhF9h0d7BhA7atasaffcc4/7/++//7Y///zTPvvsMxs2bJjb0I8aNcoOO+z/8u+YMWOsVKlSeZ72iy++mKfX9ezZ07p06WL57fvvv7c333zTOnToEBoWfNdY8fnnn9vrr7/ulsGZZ57p1kd2Bg8ebLVq1Qo93759u82bN8/Gjx9vP//8s02ZMqWA5tp/WZWNWPTKK69keP7HH3/YjBkz7KmnnnJhpl+/fvn2WSpb+rzq1atbNLz33nv27bffZhim+TnmmGMOyedNmzbNTjrpJLe9++2336xChQqH5HNw6BBmEKJwoqO9yKPNE044wR588EG34bzkkkvc8Jx2tAfj2GOPLbA1Eq0NdXY2b97s/qo2rEqVKrnOe+S6Ouuss2z37t32zDPP2PLly2Pu++HgRK5vOe+881wNhGof8jPMZLUtiLZDNT8rVqywBQsW2LPPPmu33nqrC0233HLLIfksHDo0MyFX//jHP6x8+fL28ssvZ1ulHgSd0047zc444wzr27ev/f77726cmga+/vpr9wiqroNqbE1TG2Q1q3zxxReZmplkz5499sADD1jDhg2tQYMGrrlq06ZNOTY/hFeT6xHU9uhv8NrI9+3atcvGjh1rF154oZ166qnWsmVLV9Oxb9++DJ915513uuHnnnuue91VV11lixYtynEZqqZr8uTJdvHFF7tlpPeOGDHCfaboewfLs3nz5gfcnHLkkUe6vwkJCaFhP/30k3Xv3t0tYz169epla9asyfC+CRMmhL732WefbUOGDLFt27aFxmtZTpo0yS17NYOp5kgBN5j/wDvvvOPCmF6jcKUaJNXwBdRM0KJFC/vvf//rloWax9RU8sYbb2SYzrp166x3795Wv359N50XXnghy++rGqiLLrrITUfLVNPXsg5omXbt2tUdeetz9Lp27dq5I3DJrmysXr3aNTk0btzY6tSpY1deeaV9+umnuS7/nMqqvrOW46xZszK8R7WeGq6atQMNHuHrWz766CO3HrQ+tfw0T3/99Vdo/M6dO906VtOklonW/XPPPZdjM5N+v1oOWh5alrNnz840LyoPDz/8sJ1zzjluulrHKhPh9Pt+4oknbPjw4a4c6ffQrVs3W7VqlRuvdaia38impchmph9++MGVEW1vVJOkcqvvqe+2P1Q2jjrqKDcdfa+pU6dmarYLyq3mq1GjRta0adNQuc6tDAav0fpQINP3VRl8991392s+kTPCDHKlpqUmTZq4HXZWbfPaCKvPgXb+qhUYOHCg6+9x++23h5pzVJOjh456wptHtHHQBl87Pe0As6If/ZIlS+yhhx5yr9VO4frrr8+0wciOPk/TF/3NqnlJnQ2189LR2eWXX+6q7rWBV9Na5Ovff/991wforrvuskcffdTS0tKsT58+Oc6PPlfNdQoqTz75pF1zzTUuHKhJSZ+tvzfeeGNomeTWBKaApXURPFSr88EHH7gdkjaWxx9/vHudmpwUtjZu3Oh2HgogCjLqj6BhQRB95JFH3Dzp/Qo7ana5//77M3zm448/7t6jZXLddde5dRneD2rcuHF22223uQ22dlaajpaVAkL4DkZ9eu677z4XHhQKK1eu7KajI2TRTlcBWiFM83D33Xe7nUFks8PTTz/txqlsan1p/lX+NCzc4sWL3fe66aabXFhNTEx060s7o6zKhpatwt+OHTvcjlnf6+ijj3br55dffslxveRU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AAAUTJh59tlnbcSIEXbUUUdlampKSEggzAAAgNgOM88//7z169fPunXrlv9zBAAAcKhPzd61a5e1bNnyQN4KAAAQ/TBz8cUX27///W93ijYAAIB3zUzbtm2zqVOn2owZM6xy5cqZrjfz0ksv5df8AQAA5H+YqVq1qvXo0eNA3goAABD9MBNcbwYAAMDLMDNw4MAcxw8bNuxA5wcAAKDg75q9d+9e+/nnn+2dd96xsmXL5sckAQAADl3NTHY1L7qY3k8//XQgkwQAAIhezUzgwgsvtA8//DA/JwkAAFAwYeavv/6yV1991cqUKZNfkwQAADg0zUynnHKKuwdTJN1F+4EHHjiQSQIAABRcmIm8KJ6CjS6cV716dStVqtSBzQkAADEuNXWDPf74CJs37xt3AH/BBS3shht6uf8XL/7Oxox5zFasWGbly5e3K6/8h1188aWh97799ls2efIEN42qVU+wPn1utdNOqxvV7xPXYaZRo0bu76pVq2zFihW2b98+O/744wkyAIBCS7fwueuu/la6dGkbO/YZ27p1iw0bdp8ddliiXXXVNda370122WUd7a67hti6datswICBlpSUbGee2dS+/HK2PfrocOvf/y6rWbO2vfvuDOvX72abPHmqJSenRPurxWeY2bJli7vWzMyZM+2oo46yv//+27Zv324NGza0sWPHuhUNAEBhsnr1L7ZkyXf21lvvW9mySW5Yt27dbezYx61SpUqWlJRk3bv3MvXCqFevln366ef24YfvuTDz7rvTrXXrttayZWv3vuuvv9E+/vhDmz17ll1yyWVR/mZxGmbUL2b9+vXuujInnHCCG7Z8+XIbMGCAO2176NCh+T2fAABElQLMyJGjQ0EmsH37Nmvc+EyrXv3kTO/ROOnU6Z9WokSJLO91iCidzfTxxx/bkCFDQkFG1F9m8ODBrrYGAIDCRq0OjRs3CT1XF4vXXnvV6tdvaBUqVLTatU8Njdu4caN99NEHbpycfPIpVqXKsaHxanZas2Z1aDyiEGbU0emwwzK/VR2B1eQEAEBhN27cE/bjjz/aDTf0zDB8166d1qdPH1eD065dh0zvW7v2Vxs69F7X5KSQgyiFmfPPP9/uvfdeW716dWiYOgOr+emcc87Jh9kCACC2g8yUKf+xwYPvsxNOqJ7hmmv9+t3q9omPPDLKihUrlqnfTZ8+3V0f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zxoAAChA3oYZGThwoLu+zD//+U+79957rU+fPtayZctoz1bMOv/88+21117LNFzDNC4ro0ePts6dOxfA3KGw2LNnjys3F1xwgdWuXdvOPfdcGzZsmG3bti3X9/7666928sknu7/5RdP76quv8m16iA/aJqrsBI9TTjnFGjVqZDfeeKP99ttvBbqNRiE+mymonRk+fLh7AIgNI0aMsNmzZ9sDDzxgVapUsTVr1tiDDz5ov/zyiz311FPRnj0gz3S7nDZt2rj/df2y5cuX2z333OOuNv/SSy+xJGOI12EGQOx5/fXXbejQodakSRP3vHLlyq45+JprrrENGzZYuXLloj2LQJ6ULl3aUlJSMpx4ctNNN1m/fv1s69atbjxig9fNTMhfQRX/2LFjrWHDhnbfffeFmg3uvPNOq1OnjjVv3tzeeeed0HvUdKDmPu241KSg+2Kp/1JA03vzzTetbdu2bnynTp3ckToKr4SEBPvyyy8zXIlbV+p+++233YUuI6vS1QSkchLuvffes2bNmtnpp59ugwcPDl12IasmUTWDqllLBgwY4B6XXHKJK5OrVq1yw+fOneuaoFWGb775Zvvzzz9D7585c6brf6eTB3TT2ttuu822b9/uxmm6t99+uzsa17xomrqFCuJXcD0znWgS2YQZXj41XP+r7Ojq9OPHj3flWE2uZ599tusiofGvvPJK1L5LYUKYQSbz58+3adOmhTpTf/vtt6Ef6tVXX219+/Z1TQai5oOff/7Zdb6eMWOG2xko+IRfA0g7BA3T+3U6/ahRo1jqhZjKzcSJE0Mb8vfff9927txp1atXtyOOOCJP03j11Vftsccec81Sn332mT399NN5/nyF51tuucW9p2rVqm6Y7tmmMqi/Kq/aocjq1atduFHIfvfdd13ZVBOZPj+g+dd1rVTj1K1bN9eMpmkg/qi8KJQojJQsWTLX169du9ZtC7Xt0wGd3vvf//7XbRMV2BWi77//fktLSyuQ+S/MCDPIRB2qjz322NCOQM0CaiaoVq2a25jrKEM3+ZSgBkenyuv11157rbudxMaNG0PT+9e//uWOaE866SQXhnSncxRevXr1skceecSOOeYYFwpULa+NvwLy/vRVUDlTh0uFjZdffjnP71UNi4LUaaedFhrWu3dvO+ecc1zt4F133WXTp093tYqqPdLzK664wjWHNW3a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",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "area_counts = scommerce_df['Area'].value_counts().sort_index()\n",
        "\n",
        "area_labels = {\n",
        "    1: 'Urban',\n",
        "    2: 'Suburban',\n",
        "    3: 'Rural'\n",
        "}\n",
        "\n",
        "area_counts.index = area_counts.index.map(area_labels)\n",
        "\n",
        "ax = area_counts.plot(kind='bar')\n",
        "\n",
        "for container in ax.containers:\n",
        "    ax.bar_label(container)\n",
        "\n",
        "plt.title('Distribution of Respondents by Residential Area')\n",
        "plt.xlabel('Area')\n",
        "plt.ylabel('Count')\n",
        "plt.xticks(rotation=0)\n",
        "\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "Bmx1hTmv2eiI",
      "metadata": {
        "id": "Bmx1hTmv2eiI"
      },
      "source": [
        "The bar chart shows that the majority of respondents reside in **urban areas**, accounting for **461 respondents**. This is followed by **rural areas**, with **222 respondents**, while **suburban areas** represent the smallest group, with **74 respondents**. This distribution suggests that the study captures perspectives from respondents across different residential settings, which may influence their access to technology, online platforms, and social commerce activities."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "11mldQkNDXU-",
      "metadata": {
        "id": "11mldQkNDXU-"
      },
      "source": [
        "#### Frequency of Social Media Usage\n",
        "\n",
        "Since the study focuses on social commerce, the frequency of social media use provides important context regarding respondents' exposure to social commerce platforms."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "id": "8sjukSbfESaE",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 465
        },
        "id": "8sjukSbfESaE",
        "outputId": "c3883773-2bbc-48c7-a6d5-2d53fa8943c8"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjMAAAHACAYAAABXvOnoAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjksIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvJkbTWQAAAAlwSFlzAAAPYQAAD2EBqD+naQAAUAZJREFUeJzt3Qm8VfP+//FPTpoHOufULaUoEqVRGVOkSW5Upmu4CZUmriG/bqYSEeFqoIiLSCpRcTMPdUvSqBIlcqR0jqQ0Y/8f76+79n/vffYZO+fss1qv5+OxH+fstdZew/e71nd91vf7XWuVCIVCIQMAAPCpwxK9AgAAAAeDYAYAAPgawQwAAPA1ghkAAOBrBDMAAMDXCGYAAICvEcwAAABfI5gBAAC+RjBTiIrD8wiLwzrAn9h3APhFYIOZq666yurXrx/+nHDCCda0aVPr1q2bPf/88/bbb79FTX/OOefY//3f/+V6/u+9957dfvvtOU6neWre+V1OVnbs2GGDBw+2zz77LGqb9fGTf//733bGGWfYySefbOPHj487TWQ+ep8TTzzRWrVqZb169bKVK1faoWjMmDFuWwuD0nrSpEkHNY/YfbugfP/993Hz3Pt06dKlwJcZdL/++qv17dvXGjdubKeccop9++23cafbvXu32y87d+7sjtnmzZvbZZddZtOmTSuU4Dg/5aX2Ea1jTsdVo0aN3HbHM2XKFDdNQe3fkevk7d+vvvrqQc3z1VdfdfPR/Iq6/EiEkhZgOuHdfffd7v/ff//dfvnlF/v4449t5MiRLgh47LHH7LDD/oz3xo4daxUqVMjTSTg3+vXrZ1dffbUVtC+++MJef/116969e3iYt61+oYLkwQcftDZt2rigpGbNmllO26NHD7v44ovD3/fv32/r1q2zJ5980q655hqbO3eupaamFtGa+9+//vUvGzBggBVnN9xwg9s3YpUpUyYh63Moe+211+yDDz6wu+66y4477ri4x6KCFQU8GzZssN69e7vp9u3bZ/Pnz7c777zTHY///Oc/C3S98lou54UuaN9//33761//mmncm2++aYWlatWqNnXqVDv66KMLbRmHokAHMzoImjRpEjVMkfaxxx5r9913n82ZMye8IyvwKQxFucPWq1fP/ETB5R9//GHt2rVzV4PZ+ctf/pIpL1u2bGm1atWy66+/3t5++2274oorCnmNUZR07MTmOQrH9u3b3d+//e1vVqJEibjTLFmyxBYtWmTPPPOMq031KODUReHkyZPdsViQFxWFVS5Ls2bN7D//+U+mYObHH390F7sNGjRwNeAFrVSpUuzX+RDYZqbsXHnllVatWjV7+eWXs6zO9AIdVaWeeuqpduutt7qdXNSU8+mnn7qPqvF0gOuj/zXPtm3bugPlv//9b9yq+AMHDtiIESPcCbxFixauuWrbtm3ZNhd58/eW5dX26K83bezvdNU0btw469ixo6tSbd++vU2cONEFEJHLGjp0qBuuQknTqdo4sulm7969ds8991jr1q2tYcOGbn65aaLQ9qtwVFW0moRuueUW27x5c7iK1EsXXc3ltzq0UqVK7m9kAayCWVeYp59+utueSy65xBYuXJhp3TRcTY/KB9UCfP3111HporxTzY/mo21QLdumTZui5vP555/btdde67ZPea4rV12hxuablq/aJ1Xj60Tw0EMPudrCyLxSjaHGaZ2GDBnihsVSIav9V/NRMBe77yhddQJYsWKFXXrppW77tT9G5peX1rrq9f7Pbx6LrjK17+hY+fvf/25r1qwJ54OW/8gjj0RNv2fPHpeeTzzxhB2MrI653KSTrF271nr27OnSW/uimkr03SsHsmoOiHdMv/vuu64JW9urPNTxrSaZyCr/8847zz788EO74IILXBp36NDB1YhE2rp1q1vX0047za2XtmHZsmVu3KBBg1z+RB6/ouNX88pKTuWA9nWvCUTN8Vk166Snp7u/scsXHef/+Mc/oo5DNVVpnZUeCkq1HAVEsbWz9957r5111lluGtU0K42yKpeVJ2peP/PMM+2kk05y6aTvP//8s+WVmspUqxTb1KRa3mOOOcalRayc8ll0XtCxp31P+bJgwYKo8fH2q8WLF7tyRGWR9g1tt/IkXlrnV26OcR0TqrHVOU/pq3zRNuq3HqWXyldvH1W+q6UitgzPTVrlBcFMvEQ57DCXETphx/adER1wOkB00D/11FPuxPLJJ5+4k7HXnKMThj4qyJXpHp0gVBgps5XR8ehqYPXq1fbAAw+4aXXw6oom8uSWHS1P8xf9jde85FUJP/300655Ridl7bxqWoud/q233nJ9gO644w534snIyLCBAweG1+f+++93zXNaV+385557ro0aNcpmzJiR5TqqkNbJu3r16m6eSkMVyjrIf/rpJ3fyU1qJAgmlY3Z0UCuvvM+uXbts6dKlNmzYMKtYsaJbJ6/g1glV26ODTMtQrc51110XDmjS0tJcYKIDWidU1dJ98803ruo8svDQPFTgKF20HDXtqUDWyVi0T1x++eXhNNLBqmBNwWBkYCQKhnUCVz6oz4fyRSdPz2233WavvPKK9enTx+WRaq1imzJV4Olkq2YWTaMgUAWnAtrIwkbbcNNNN7nCWictneSVX/PmzXPjvbRW0533f37yWLZs2eLSWMtTPmu9lUY//PCDHXHEEa7Wbfbs2VH9Kd555x1XqF144YV5ynN94h0jscdcbtJJ662aPF15K7BUAa6mNx2XeaXt69+/v6vxVdCgec2aNcvtY5HbrWBg+PDhbj2UL2rK0Xp7+4r2ae1PCtK0P2i7Spcu7Y4jBQbKL11QabxH26OT70UXXRR33XJTDuiv5i3aH7Te8SgoLFeunN18880uzbQeXnrWqVPHlWEpKSnu+/r1692JTCduHT8PP/ywC3R0bCovRHmpbVP6ab9XPy6lodIysi+gR8ed0k7ppXXWfqrvb7zxhj366KN5zjcFGloHNTXFNjGdf/75+cpn7T/aJpVJjz/+uFs/pVd2vKBax4u2Q2WSLnKV/zpXFJT7czjGFUjrmFA669ykc5/S4YUXXnD9TD3aXq2XzhFaX+23o0ePznNa5VkooK688kr3ycqoUaNCxx9/fCg9Pd19b9u2bej22293/0+YMCHUtGnT0L59+8LTf/jhh6ExY8aE/vjjj7jz/+STT9z8xo0bF7UczVPz9uj/008/PbRr167wsHfeecf99v33389y3b3562+877G/0/pq/Jw5c6Lmo/XT8K+++ir8m8aNG4d27twZnmbmzJlums8//9x979ChQ+iOO+6Ims/YsWNDH3zwQdy0/f3330NnnHFGqFevXlHDN27cGDrppJNCDz74oPuelpbmljNjxoxQdjRNvE/Dhg1DPXv2DK1ZsyY87dSpU9245cuXh4cpz6644opQt27d3HeliabZsmVLeJoVK1aEHnnkkXA6KF20rt999114mtWrV7vfvfTSS+57jx49Qp07dw799ttv4Wl++eWXUMuWLUODBg2KyqdHH300apvOOeecUJ8+fdz/yovI+XppqHlruOfSSy8NdenSJWp5GzZsCDVo0CA0efJk911pqd+8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GB62atUqd5u1OvuuXr3a9u7dGx6nDsJ6pozor757VIuzZs2a8HgAABAchyWyaUl3IP3zn/+09evX20cffWQPPfSQ9e3b193RVL16dRsyZIitW7fOJk6caCtXrrQePXq436oZaunSpW64xmu6mjVrcls2AAABlLBgJikpycaPH+9uq7700ktt6NChdtVVV7mn+XrjdNeSHow3a9YsGzdunNWoUcP9VoHLmDFj3HNnFOBs377djS/BbRwAAAROiVBAHpubnp64zqinjP7YgmjxLa0TvQqBolheHeXU8ToYR3Wwkd/BEuT8Tk3NuQMw9yoDAABfI5gBAAC+RjADAAB8jWAGAAD4GsEMAADwNYIZAADgawQzAADA1whmAACArxHMAAAAXyOYAQAAvkYwAwAAfI1gBgAA+BrBDAAA8DWCGQAA4GsEMwAAwNcIZgAAgK8RzAAAAF8jmAEAAL5GMAMAAHyNYAYAAPgawQwAAPA1ghkAAOBrBDMAAMDXCGYAAICvEcwAAABfI5gBAAC+RjADAAB8jWAGAAD4GsEMAADwNYIZAADgawQzAADA1whmAACArxHMAAAAXyOYAQAAvkYwAwAAfI1gBgAA+FpCg5kff/zRBg0aZC1btrSzzjrLRo4cafv27XPj0tLSrGfPntakSRPr3LmzzZ8/P+q3CxYssC5duljjxo3t6quvdtMDAIDgSVgwEwqFXCCzZ88ee/HFF+3RRx+1Dz74wB577DE3rn///paSkmIzZsywrl272oABA+yHH35wv9Vfje/WrZtNnz7dqlSpYv369XO/AwAAwVIyUQvesGGDLV++3P773/+6oEUU3Dz44IPWunVrV9Py8ssvW7ly5axu3bq2cOFCF9gMHDjQpk2bZg0bNrRevXq536lG54wzzrBPP/3UWrVqlahNAgAAQaqZSU1NtaeffjocyHh+/fVXW7FihZ144okukPE0b97cBT+i8S1atAiPK1u2rJ100knh8QAAIDgSVjNTqVIl10/G88cff9jkyZPt1FNPtfT0dKtatWrU9MnJybZlyxb3f07js1KiRIFuAnJAeicmvUn3YCC/g4X8LqbBTKyHHnrI1qxZ4/rA/Pvf/7ZSpUpFjdf3/fv3u//Vzya78fFUqVLekpK4easopaRULNLl4U/JyaR7kJDfwUJ+F+NgRoHMc8895zoBH3/88Va6dGnb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18FMkFSsWNFSU1Ndx+fjjz/eBg8ebOeff76NHDkyx9/qlnX9FsWDHu749ddf265du8LDFMRce+21LnjZu3dveLgeAkkw4//gNTaY+fXXX91F2Iknnpiw9ULuylyv3D3jjDNs0KBB7ljduXMnyVfMEMz42KWXXmpfffWVbdy40davX+9OhqqFUSfov/3tb+6EmV2V8xNPPGEXXHBB1DD1O9JvUXiUP7paX716tfu+ZcsW1wxx8cUXuwLUayr45ptv7JdffnEnQ+WzmgVOPvlk69Chg3ttR6R33nnHOnfubI0bN7YePXrYp59+GnfZ3333nZ1++un2+OOPRw1X53mdWLdt2xYepveeaX468SL/zj33XJcfken44YcfunwtX7581LRqYujUqZPL527dutnixYujmiCU75dcconbh9T8qDwS7RubNm1yzZeRVf9qyjrrrLNcuaBx8Z7Bpab5Z599NmqYygW9LgYW91lm3k0mOZW7yrO7777bXcBMnDjRDX/55ZfdcP1G+fbll19mSubCOh4XLlzo9hutq/ZLrYtnx44ddtttt7laf724+d577426sPrss89cP1XtmzfeeKN71ltxQjDjY3Xr1g0fUH379rWjjjrKXn/9dbeD/v777/bQQw9l+3vV7OgkqZOm5z//+Y8bjsItEFUoqeO6fPLJJ9awYUN3YjvllFPCTU2qpTnuuONcc+r111/vCsRZs2a5am71pXnttdfCr/bQMLXna/xf//pXN72C3EgqGFXw6mSpK8xIKsB09amgKHJfOPvss61ChQrsDgdBNalK248//jg8TOncrl27TIGMTiBqPlbeKuhU/6off/wxqq+LhimfFfiOGDEiPPwvf/mLaxoZOnRoePq33nrLJk2aZGPHjrW5c+fajBkzMq2fjndN59HJWGXCodj/8GDoQkABiYJDHat6bU5O5a4CTAWQytsuXbq4GjrlxZ133mkzZ850x7RuWtFFS2Efj7///rvddNNN7t2FmpcCEvU11flDtN+oxkkBsMoXNYUPHz48XHZov9Q+qX2zXr16bn8qTghmfEyFmai5Qh2CdUV29NFHu07RF110UXgnzYqmVZTt7ZQ68NasWcOLOouArsq9YEbBS6tWrdz/alKKDGb0ffbs2ZacnOwKojp16rirOhWizz//vJtOJytdretqunbt2q5wbN26tSuUIl/MqpOg8vuOO+7ItD7qu6GancgCSv8T2BYMXQV7TU06uf33v/91wyK98MIL7kpdV7/HHnus3XrrrS4Qmjx5cngaHdcKgo455hi75pprwjUzRxxxhCUlJbkywSsXRLUCmoeaSHQiUuAbSydZ7WuqIRSd6HRlXrlyZQsypZ1qT7xaF+WLLiC9YEW1Frkpd6+77jp3XNaoUcOefvppFxS0bdvWHcs6phUMKTgt7ONx586d7lU/eu1PzZo13UWPauTUjKZATX24tG2qxVc5ocBaAZd+p31CrwhSzY32Td1sozQpTnz7OgP82e4uitRVMCpiVuG2YcMGF5Rop82JDg7tsLqq1w6rk6dOnCj8YMarWVHwooJDlP4PPPCAO+HpBKN8Ub8ZnYQiX6qqqyydvLwraeWdOoZ7Dhw44E5IkSdKPWBSQVNWnU51Uvv3v//tHkaZlpbm/rZp06bQ0iBIdHyqNkx5oKp+BRixx5nyUZ14IzVp0iTcbCE6AXp03Cufs6OTrEdBTrxmJp2gdQLTyVI3UWhf0gk36JRfqp3SxaJqvnSxd8stt7h3Akq5cuXs8ssvz7HcVeDgUV4qYHjkkUfCw/Q+wXgdt/NyPJYsWTJu3oZCIfdXzdoKeLW+uphRzYsCqu7du7ugVU3bqmnSRVAkDfO6MeiOrsiyQ8FMcWpqIpjxMa+tVZG9+knoINNVuw4CHVjq/5ITRf+6I0w7rKqadYWPwqfAZOvWra4qV39VrSxqVtJJR30lVIAouFH7+WmnnWZ33XVX3HkpsFGzkq4cI0W+1kNXjTpR6cpKfTG8JsrYB1Hq5KcrNBWuOgHrtSE4eGpOEOWl0ve8887LNE28tFbe6oTi0UkpL7yAN/bkFu+i5u2333ZNKLqNN7bWKIgUbKpGRf71r3+5MrZfv37uokH5oCAnN+VuZL4qP9UUqOM5Urymo7wcjxUrVowbEKkfjDde9Fy2K664ws1TH22LAhutl6aJ1wyp5q54+47SoDgFMzQz+Zh2PJ2k1HlUJ0Q1O6hKU9XJGpZVwRVJtwPrhKl56eqfdvKioas6FVYqTHSFo34xoisf9ZtRG7uuwlW1qyYF9WHQFZ4KV31Ua6PaFtF4nYC8cfpovpF9NFRLo74yKkS9dvB4VCB/8MEH7lZUmpgKjq6c1d9BTU1K39j+Ml4+qhYukr5reGFTvmtZqmXQesZ2TA469XNT/yQ9y0e1JaJO3Xktd5WXas6LPFaffPLJqCfZ5+d4rF+/vqsdiq2pU56qHFF5o9f9qI+MlqkaX5X5p556qtsntV5qTlL5462XmtFGjRrlanx0kaVaJwU9HqVFcUIw4xPa0bQz6uBRjcx9991nb775pmuvVfWh+kQo0tZJTXch6K6HeNWO8XjVmWpXD3o7eVFS0PLGG29kuvVa39977z03XtS2rYJFNTOqplbBpvz3milU46J9QYWq2r6Vl/pENkl4dFWo2gEtN6t9Qc/V0L6m/QEFR1fWOjaVb7Vq1co0Xvmo/jEKKBS8Pvzww+4CQ1f/uaETlmoG1C8ir9SfQ/0knnvuOYLYLCh9lBeqyVCn7PyUu+rnpDRWHutYVZOTmvXi1ZTm5Xg877zzXCCiR3Zon1FNu5ahGiXv+Wsq29Wh+P7773fLVu2vptVdU1q+auXUT0t9+XSnpe5+0/bpXYcKpFQLo3JH+5j6/qgcKU5oZvIJ7YD6aIfV1bp2QJ2w1PdC1NauqFvtr4rSdeJT7/TIOyGyotoYVT+qyQlF2/SgKmmv829kMKOCwwtyVAX91FNPufxXU5IKUVUVe/0a1K9CV1Bq19dfVU2PHj06HAxF0hWYOpmqX46uwGPpikx3Kmj/ymuTBrKn2jH1mYlXKyM6/jIyMtxt8zp5qeZO+0dWJ7pY6g+hAEjNDfl5uq+Wrwsl+kll7R//+IdrjlcQorTOa7kbmcf6q2NNj8iId+GRl+OxfPnyLhDWeil4URCickB9fLyuA6pdUiCmckQXSPqNgjM9EkJUdqj2Sb9XTaKCG+9mAQVCCmB0ntCt3Spb9Dc3tf9FpUSoOK0NEkKFn06SusOC6uVgU/8MdQxUPypVQSM49IRbNYEo71E8cDzmHjUzAb8bSlWY6l+hakQCmWDTg9y0P6jjME8dDg41Naj/w0svveRqCVA8cDzmDcFMwKkaUdWROT1gD4c+Pa9GfTUee+yx8BNOcehTx1E1L+jptV6zNRKP4zFvaGYCAAC+xuUXAADwNYIZAADgawQzAADA1whmAACArxHMAAGid8joVQmxNEzjihutkx5GFvvRA+IAwMOt2QCKNb2CIfbp1DydGEAkghkAxZre5puampro1QBQjNHMBCATvbRSrzXQG727detmn332WXicXoKp119onB6ydvPNN9uuXbvC42fNmuXeP9S4cWP3bhiN13ujRG9PGTdunHtPkX7bt29f96bh/NI7iO699173Eke9U0hPtd68ebObr5avZqqxY8dGve1XL9vr0KGDe6eVan30cj1v/fTiVn0iqVlr0aJF7n+9RFAPmNP7tPTRb70XO+plg5r27bffdtuv9NH7syJf/Kg3mV900UVu3fR+nIULF7qXiDZr1sz9zqO3H2v+Gg8gZwQzAKKsWbPGvXTu7rvvdm/0VdBx0003uffE6G27N954o3tarMbpacELFiywV155xf1WQY8ChOuuu871wylbtqx7o7dHL8ObPXu2exGmXqOhN0j36tXLnbzzS8vRE6wVtOiVHAMGDHDznTlzpo0cOdIt78knn3TT6kWKWv/LLrvMZsyY4YKruXPn5npZjzzyiHtirl78qYBPwZPmF0nL0nTa1s8//9yeffZZN3zdunV2ww03uDccv/766+6NyP369bOdO3e64EcvMPQoTfWyP14rAeQOzUwAomzatMm9nb1GjRpWs2ZNF8iolkbBjD56BYb3Jl6NP/30092JWqZMmeL6tyhYEL1lV+978ujNuwqSvDeFDx8+3NXSzJs3L8sOyJpetS+R9FLUcuXKuf9VI6OaDVFNhmp6pk2b5l7JcOyxx9rtt99uQ4YMcW84VuCjN/5ec801bnq98VjLzg29yVwBioIg1cCIgj5ti4Ik791mgwYNspNPPtn9f8EFF7iARqZPn+7WUwGM9O7d273deMeOHe7daHojs96+XLp0aRdgdezY0ZKSktg7gVwgmAECRFf7CkhiaZjGiYKL448/3p2ITzzxRNeEc/HFF7vxderUsVKlSrkXEiqA0Wf9+vXWtWtX91ud1C+99NKo5TVs2ND9r6YovZVZJ+3Idz+pmUVvbs+KgoP27dtHDVONj+eoo44K///111+7Zp3mzZtHbZuW8fPPP9uGDRusQYMG4XHaFm/9cpKWluZqkLxALXL+Wv+TTjrJfa9du3Z4XIUKFcK1TnrvlTeNR4Gi9xutiwKrs88+2959991wbRKAnBHMAAHrTKumkVhq6tA4L1BQzcann35qH3zwgavNUI2L/iog0G3RqkVR81PPnj3tueeeC89HNQlquonkfff6rfzrX/+yY445JmqaypUrZ7nOajKKDBBiqSbD89tvv7namPHjx8fddm1b7PopiPCoRipyvObn8dZfb5f2aoUi19HrG5PVnVZesJjVOPXjUVOTfq8gyKttApAz+swAAaLmkWXLlmUavmLFClcLIxo/YcIEO/XUU13zjJo81PyxZMkS19dDzTTq86J+M2pO2bhxYzgAqFevnq1evToqAPjiiy/c/5UqVXIn/fT0dBec6FO9enXX30W1FgVBQZKamapUqRJehjrmPv744y5QqVu3brjZR7Teqk3yKJCI7Mys2hhPrVq1XLCmoMWbt4IO9cv56aefclw3Tb927dqoYarleeONN9z/qglTB+H333/fNTFpfQHkDsEMECCqVdHdSGomUhCiE7k6zqoG5oorrnDTlClTxt1xpNoZBQI62apvhwKhI444wv1m5cqVLgB54IEHXHCgu3zkyiuvdNPrt2rSuf/++8N9cEQ1Oeo0rBO2mmbU/2bp0qWuNqUgqIlMzU633XabW091SL7zzjtdjYwCEW2/givV3Gj91OclsolLdyCpP4763nz11VeuT49X06LARc1t6geku5vUvDZ48GCXjuo7lJu01/qoQ7B+o4BRzXSq4RI1jWk91XFZfWgA5B7NTECA6GStk6iClYkTJ7ogQzUy6ph7wgknuGnUp+S+++5zJ3ydzNURWLUnqtVQTYrudlJQouYd1dKoY61Xu9C0aVPXYVfzV5OUahg0zAsIrr32Wlfzcdddd7nmLvVXmTRpUrbNTHmhgEWBmjoMq5OymoO0DuoELFWrVnXjdXu1+qQoaND6edT3R8GVOumqWUp3Kinw8Oi27QcffND141FfGG2/0jE3HXWPPvpodwu4arV0t9Nxxx3n1qFatWpuvPJC66pAL7f9eAD8qUQotgEZAPJJNTaqwYisaVHAoCBGz6spjvSsGt0CPXDgwESvinsuj5qjFCwByD1qZgAUGPW30e3Lqr3QU3tVY6OH2J111lmkcjaWL1/u+hqpCXDOnDmkFZBHBDMACoz63aifjWo5dIeUmqz0gDleR5A93ZL9zDPPuNvWc9P/BkA0mpkAAICvcTcTAADwNYIZAADgawQzAADA1whmAACArxHMAAAAXyOYAQAAvkYwAwAAfI1gBgAA+BrBDAAAMD/7f0VDYvPaZrWCAAAAAElFTkSuQmCC",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "freq_counts = scommerce_df['Frequently'].value_counts().sort_index()\n",
        "\n",
        "frequency_labels = {\n",
        "    1: 'Daily',\n",
        "    2: 'Weekly',\n",
        "    3: 'Monthly',\n",
        "    4: 'Rarely Used'\n",
        "}\n",
        "\n",
        "freq_counts.index = freq_counts.index.map(frequency_labels)\n",
        "\n",
        "ax = freq_counts.plot(kind='bar')\n",
        "\n",
        "for container in ax.containers:\n",
        "    ax.bar_label(container)\n",
        "\n",
        "plt.title('Distributions of Respondents by Frequency of Social Media Usage')\n",
        "plt.xlabel('Usage Frequency')\n",
        "plt.ylabel('Count')\n",
        "plt.xticks(rotation=0)\n",
        "\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "s7DoJqgN3RL5",
      "metadata": {
        "id": "s7DoJqgN3RL5"
      },
      "source": [
        "The bar chart reveals that the majority of the respondents use social media **daily**, accounting for **725 respondents**. In contrast, only **14 respondents** reported using social media **weekly**, **7 respondents** reported **monthly** usage, and **11 respondents** indicated that they **rarely** use social media. The results indicate that the sample is highly engaged with social media, making it appropriate for investigating social commerce-related behaviors and perceptions."
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
        "The demographic profile of the respondents indicates that the sample primarily consists of university students with characteristics that align with the target population of the study. In terms of gender, the sample is composed of respondents from different gender categories, with Females representing the largest proportion. Regarding income, the majority of respondents reported earning less than $100, reflecting the limited financial resources typically associated with university students who may rely on parental support, allowances, or part-time employment. With respect to residential area, most respondents reside in urban areas, although a substantial number come from rural communities, providing representation from diverse living environments. Finally, social media usage is highly prevalent among the respondents, with the vast majority reporting daily use of social media platforms. Collectively, these demographic characteristics suggest that the sample is composed of digitally active university students who are well-positioned to provide insights into social commerce behaviors and perceptions."
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
        "To understand how each construct is distributed among respondents, descriptive statistics and histograms are generated for the composite scores. The descriptive statistics provide information about the central tendency and spread of the data, while the histograms help visualize the shape of each distribution. This analysis allows us to identify whether respondents generally reported high or low levels of each construct and whether the scores are approximately normally distributed."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 32,
      "id": "M5R5sFNvkODv",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "M5R5sFNvkODv",
        "outputId": "c13753ef-6cce-4dfc-8df4-5bee6cd1834f"
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
          "execution_count": 32,
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
        "Across all constructs, the mean scores ranged from 3.56 to 3.77, indicating that respondents generally reported moderate to high levels of the measured constructs."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 33,
      "id": "LU-SW6JblNfT",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "LU-SW6JblNfT",
        "outputId": "aac064a0-1750-417c-d23b-30372b4bfb9b"
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
        "The histograms of all composite constructs show similarities in terms of their distributional patterns, where responses are concentrated around scores of 3 and 4. In particular, the distributions show a secondary peak near 3 and a more prominent peak near 4, indicating that respondents generally reported moderate to high levels across all constructs. This pattern suggests that while some respondents provided neutral evaluations, a larger proportion tended to agree with the survey statements, resulting in higher composite scores. Overall, the distributions are not perfectly normal and show characteristics of a bimodal distribution, although the exact frequencies are different across constructs."
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
        "The dataset measures several behavioral and perception constructs associated with social commerce adoption. These constructs include Perceived Usefulness (`PU`), Perceived Ease of Use (`PEU`), Familiarity with Social Commerce (`FSC`), Social Presence (`SP`), Trust in Platform (`TP`), Interaction Behavior (`IB`), Actual Usage Behavior (`AUB`).\n",
        "\n",
        "Understanding how these constructs relate to one another can provide insights into whether positive perceptions of social commerce platforms are associated with greater trust, familiarity, and user engagement. To investigate these relationships, composite scores were calculated by averaging the survey items belonging to each construct. Pearson correlation analysis was then performed on the resulting construct scores."
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
      "execution_count": 34,
      "id": "8d9eQ7Gct8JD",
      "metadata": {
        "id": "8d9eQ7Gct8JD"
      },
      "outputs": [],
      "source": [
        "scommerce_df['PU'] = scommerce_df[['PU1','PU2','PU3','PU4']].mean(axis=1)\n",
        "scommerce_df['PEU'] = scommerce_df[['PEU1','PEU2','PEU3']].mean(axis=1)\n",
        "scommerce_df['FSC'] = scommerce_df[['FSC1','FSC2','FSC3']].mean(axis=1)\n",
        "scommerce_df['SP'] = scommerce_df[['SP1','SP2','SP3']].mean(axis=1)\n",
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
        "Correlation Matrix\n",
        "\n",
        "Pearson correlation coefficients were computed to measure the strength and direction of the relationships between the constructs.\n",
        "\n",
        "The resulting table shows the correlation coefficient between every pair of constructs."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 35,
      "id": "OV-8VM9NvALc",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OV-8VM9NvALc",
        "outputId": "f9f17c8a-c1e8-477c-b5a5-4077c53f178a"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "           PU       PEU       FSC        SP        TP        IB       AUB\n",
            "PU   1.000000  0.788226  0.804070  0.713556  0.676910  0.655539  0.771895\n",
            "PEU  0.788226  1.000000  0.776458  0.712217  0.674808  0.691334  0.785858\n",
            "FSC  0.804070  0.776458  1.000000  0.757518  0.710337  0.657147  0.744687\n",
            "SP   0.713556  0.712217  0.757518  1.000000  0.724268  0.627348  0.682869\n",
            "TP   0.676910  0.674808  0.710337  0.724268  1.000000  0.663131  0.662269\n",
            "IB   0.655539  0.691334  0.657147  0.627348  0.663131  1.000000  0.683213\n",
            "AUB  0.771895  0.785858  0.744687  0.682869  0.662269  0.683213  1.000000\n"
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
      "execution_count": 36,
      "id": "PYbIAzvcvLMJ",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 538
        },
        "id": "PYbIAzvcvLMJ",
        "outputId": "4480b798-f09e-4d40-84c0-c8fbad376aa8"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAnUAAAIJCAYAAADDF8vuAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjksIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvJkbTWQAAAAlwSFlzAAAPYQAAD2EBqD+naQAAuzZJREFUeJzs3Qd4U+XbBvC7TZN075ayl+y9QcCFeyAgoihDGe69t6LyV3ErfgIKKkNUkCFLBAURUGQIsveG7r3SrO963pC0aVNataU54f5dV64255ymyZtz3vOc5x3Hz26320FEREREmuZf02+AiIiIiP47BnVEREREPoBBHREREZEPYFBHRERE5AMY1BERERH5AAZ1RERERD6AQR0RERGRD2BQR0REROQDGNT5IF+dT9pXPxdRTeOx5R3lwu+B/qvzOqjbvn07nnzySVxyySVo3749Lr/8crz44os4fvx4Tb81bNiwAS1atFA/K6uoqAj/+9//sGjRIteyZ555BpdddhnOBflf8p4vuuiiciund955R20zfPjwf/Ta+/fvx9ChQyvcbt68eer1T5w4ger25Zdfonfv3mrf+b//+z+P28h7Kflo3bo1LrzwQjz22GM4efLkP/6fUm7/tOz+Ldlv5Ds9F+T7kvKR76+mVFS2zvdY8tGyZUt06tQJgwYNwty5c6E1iYmJuOuuu9z2xXP5vTv9/vvveOCBB9C3b1906NABV111Fd566y2kpaWhJniqS8/190D0bwTgPDVr1ix10Pbo0QOPP/444uPjcfToUUydOhU//fQTvvrqK1Vha0lycrJ632+88YZr2X333YcRI0acs/fg7++PpKQkbNmyBV26dCmzfunSpf/qdX/88Uf89ddfFW4nAfq3336rvs/qlJubq0468v9GjRqFevXqlbvt4MGDcfPNN6vfzWazqrg//fRT3HHHHViyZAkMBgO80cSJExEaGlrTb8Pr3Hvvvep7F3LxkpeXhzlz5uD555+HxWLBrbfeCq1Yv349fv311xr93uVC7/PPP8fVV1+tyjAyMhJ79+7FZ599purimTNnonbt2qjpuvRcfw9E/8Z5GdRt3rwZ48ePx+23364qEScJ8CRbN2DAADz33HM1mjWoKg0aNDin/08qXznRLVu2rExQt3XrVhXwNW/evNr+f3R0tHpUt6ysLNhsNrW/dOvW7azb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IC6++GJkZ2erv4mPj3frwhYZGYnExMQqfc+VDup++eWXKv3HRERERN6qoKDALaATzuelxxNIUCdTvslgirvuuku1YsrAiW+//VYFgiX/tuRrVfW4BPapIyIiIu93jvvUGT0MBnU+DwwMdFt+3333Yfjw4a6BES1btsTOnTvx3XffqanfSv5tydeqyv50gtMzExERkdfz8/evtocntWrVQkZGhupXV7JJVgK68PBwt22lf50zoHNq0qQJkpKSVDOrBIipqamudfKa0oTraVDFf8GgjoiIiKiUVq1aqb5vW7dudS3bvHmzGuFacpCEeOaZZ9Q0JSXJIAgJ7GRb+Rv5Wyd5TXltyehVJQZ1RERE5P38/Krv4YE0jcrAz1deeQV///03Vq5cqQaMjhgxwpW1k350zhsyLFq0CAsWLMDRo0fVnHQSxA0bNkytv+222zB16lT1GvJa8ppDhgyp8uZX9qkjIiIi8kCybxKAyRxzMrpVRrReeeWVap0MinjjjTcwaNAgtezll1/Gp59+qiYpbtasGT7//HPUq1dPbXvdddfh5MmTarJh6Usn28scdVXNz26323GO7b3lqnP9LzXjwLwjNf0WvFazwY1r+i14paCokJp+C14rIMhY02/Ba9ks1pp+C14rNzGzpt+C12rx7fIa+9/5X46rttcOvuNl+AI2vxIRERH5ADa/EhERkfcrp+8bFWOmjoiIiMgHMFNHREREXq+8+eSoGIM6IiIi8n7n+N6vWsQSIiIiIvIBzNQRERGR9zvH937VImbqiIiIiHwAM3VERETk9fzYp65CzNQRERER+YAaydTZLLaa+LeawFthlW//3MPn8JvQjla3N6vpt+C1OAVC+fQhgefwm9AWQwhvL+eV2KeuQszUEREREfkA9qkjIiIi78c+dRViUEdERETej/d+rRCbX4mIiIh8ADN1RERE5P1479cKMVNHRERE5AOYqSMiIiLvx4ESFWKmjoiIiMgHMFNHRERE3o+TD1eImToiIiIiH8BMHREREXk/9qmrEDN1RERERD6AmToiIiLyfryjRIUY1BEREZH34+TDFWLzKxEREZEPYKaOiIiIvB+bXyvETB0RERGRD2CmjoiIiLwfpzSpEDN1RERERD6AmToiIiLyfhz9WiFm6oiIiIh8QJVk6kwmE4xGY1W8FBEREVFZHP1atZm6v/76C7fd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",
            "text/plain": [
              "<Figure size 800x600 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
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
        "The correlation matrix reveals that all behavioral and perception constructs exhibit **positive correlations**, with coefficients ranging from **0.63 to 0.80**. This indicates that respondents who score highly on one construct generally tend to score highly on the others as well.\n",
        "\n",
        "The **strongest relationship** was observed between **Perceived Usefulness (PU)** and **Familiarity with Social Commerce (FSC)** (*r* = 0.80). This suggests that respondents who are more familiar with social commerce platforms are also more likely to perceive them as useful. A similarly strong relationship was found between **Perceived Usefulness (PU)** and **Perceived Ease of Use (PEU)** (*r* = 0.79), indicating that platforms that are easier to use are often perceived as more useful.\n",
        "\n",
        "Other notable positive relationships include:\n",
        "\n",
        "* **PEU and FSC** (*r* = 0.78)\n",
        "* **FSC and Social Presence (SP)** (*r* = 0.76)\n",
        "* **SP and Trust in Platform (TP)** (*r* = 0.72)\n",
        "* **FSC and TP** (*r* = 0.71)\n",
        "\n",
        "These findings suggest that familiarity, ease of use, usefulness, social interaction, and trust are closely interconnected aspects of social commerce behavior.\n",
        "\n",
        "The **weakest relationship** was observed between **Social Presence (SP)** and **Interaction Behavior (IB)** (*r* = 0.63). Although this is the lowest correlation in the matrix, it still represents a moderate positive relationship, indicating that users who experience greater social presence are somewhat more likely to engage in interaction behaviors.\n",
        "\n",
        "Overall, the results suggest that the constructs do not operate independently. Instead, positive perceptions of social commerce platforms tend to occur together with higher levels of trust, familiarity, social presence, and interaction behavior. This supports the idea that improving one aspect of the user experience may positively influence other aspects of social commerce engagement."
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
        "As the potential target variable, it is also important to examine how each behavioral and perception construct relates to Actual Usage Behavior (AUB). The correlation matrix was therefore extended to include AUB, allowing the strength of its relationship with each construct to be compared."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "FdOo28znV_VI",
      "metadata": {
        "id": "FdOo28znV_VI"
      },
      "source": [
        "The correlation analysis indicates that all behavioral and perception constructs exhibit positive relationships with **Actual Usage Behavior (AUB)**, with correlation coefficients ranging from **0.66 to 0.79**. Among the constructs, **Perceived Ease of Use (PEU)** demonstrated the strongest positive correlation with AUB **(r = 0.79)**, followed by **Perceived Usefulness (PU) (r = 0.77)** and **Familiarity with Social Commerce (FSC) (r = 0.74)**. These findings suggest that respondents who perceive social commerce platforms as easier to use, more useful, and are more familiar with them tend to report higher levels of actual usage.\n",
        "\n",
        "In contrast, **Trust in Platform (TP)** exhibited the weakest positive correlation with AUB **(r = 0.66)**, although the relationship remains moderately positive. Overall, the results indicate that all six constructs are positively associated with Actual Usage Behavior, supporting their inclusion as potential predictor variables in the subsequent machine learning models."
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
        "This analysis examines the distribution of Actual Usage Behavior (AUB) across different demographic categories, including gender, income level, residential area, and social media usage frequency. Descriptive statistics and visualizations are used to identify potential patterns and differences among groups.\n"
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
        "The distribution of Actual Usage Behavior (AUB) across gender categories is examined using descriptive statistics and boxplots. As only two respondents belonged to the \"Different\" gender category, this group was excluded from further analysis due to insufficient sample size. Consequently, the analysis focuses on male and female respondents."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 37,
      "id": "Fsi9ZZFgVE1P",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 143
        },
        "id": "Fsi9ZZFgVE1P",
        "outputId": "83f73b1f-422a-4058-cc5d-ad80fa22deef"
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
          "execution_count": 37,
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
      "execution_count": 38,
      "id": "d_kW-IY5tt9M",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 465
        },
        "id": "d_kW-IY5tt9M",
        "outputId": "7b5f922c-7376-41f4-9099-71bd87a28e4b"
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
        "Both the table and figure above present the distribution of Actual Usage Behavior (AUB) across male and female respondents. Female respondents showed a slightly higher mean AUB score (M = 3.69, SD = 0.67) than male respondents (M = 3.66, SD = 0.79), while the median AUB score was also higher among females (Median = 4.00) compared to males (Median = 3.75). Nevertheless, the difference between the two groups appears minimal. The boxplot further supports this observation, as both genders exhibit highly similar distributions and central tendencies."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "IfPnJe4TLvkf",
      "metadata": {
        "id": "IfPnJe4TLvkf"
      },
      "source": [
        "While the descriptive statistics and boxplot suggest only minor differences in Actual Usage Behavior (AUB) between male and female respondents, it is not sufficient to infer significant difference. Therefore, an independent samples t-test was conducted to assess whether the mean AUB scores differ significantly between the two gender groups.\n",
        "\n",
        "For this, the null hypothesis states that there is no significant difference in the mean AUB scores of male and female respondents. A significance level of α = 0.05 is adopted for all inferential statistical tests conducted in this study.\n",
        "\n",
        "- $H_0: \\mu_{Male} = \\mu_{Female}$\n",
        "- $H_1: \\mu_{Male} \\ne \\mu_{Female}$"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 39,
      "id": "aRphGz-u3dDw",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aRphGz-u3dDw",
        "outputId": "9abb8706-81dc-4f12-90a5-2ad3af3e4f7a"
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
        "An independent samples t-test was conducted to determine whether Actual Usage Behavior (AUB) differed significantly between male and female respondents. The results indicated no statistically significant difference in mean AUB scores between the two groups, t = -0.417, p = 0.677. Since the p-value exceeds the significance level of 0.05, the null hypothesis cannot be rejected. The findings suggest that AUB does not differ significantly between male and female respondents."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "ny_bBweRB-td",
      "metadata": {
        "id": "ny_bBweRB-td"
      },
      "source": [
        "#### 2. AUB Across Income\n",
        "\n",
        "This section explores how Actual Usage Behavior (AUB) varies across income groups. Descriptive statistics and visualizations are used to identify any noticeable differences in AUB among respondents with different income levels. Before proceeding, the distribution of respondents across income categories is examined, as substantial differences in group sizes may affect the interpretation of the results.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 40,
      "id": "hoPcwfUBFBkC",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 673
        },
        "id": "hoPcwfUBFBkC",
        "outputId": "554e0873-93ef-40c3-f48b-5b0a9ea10e4e"
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
        "The income groups are highly imbalanced, with most respondents belonging to the lowest income category. Additionally, the highest income groups contain very few respondents. The histograms also show that the distributions do not closely resemble a normal distribution, particularly for the smallest groups. As a result, formal inferential tests such as one-way ANOVA were not performed, and the analysis focuses on descriptive comparisons instead."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "1SwLqolaICAe",
      "metadata": {
        "id": "1SwLqolaICAe"
      },
      "source": [
        "#### 3. AUB Across Area\n",
        "\n",
        "This section explores how Actual Usage Behavior (AUB) varies across different areas of residence. Descriptive statistics and visualizations are used to identify potential differences in AUB among respondents living in urban, suburban, and rural areas."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 41,
      "id": "KCj0iTRwNQHk",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 175
        },
        "id": "KCj0iTRwNQHk",
        "outputId": "8ccf18a9-c3f0-4c80-9a2b-a814d1a8736f"
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
          "execution_count": 41,
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
        "The above presents the descriptive statistics of AUB across areas of residence. To better visualize the distribution of AUB among respondents from urban, suburban, and rural areas, a boxplot is presented below."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 42,
      "id": "-7WfN3unNfnJ",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 465
        },
        "id": "-7WfN3unNfnJ",
        "outputId": "7d76c4f5-cbc0-418c-965e-18ee37f20538"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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Ww4gsFTtr6YZBRf1bSC4ALGBmhcAABAo1LwAAIBAIbwAAIBAIbwAAIBAIbwAAIBAKbI3qUtJyf8P0wEAgOhQufJ/716eF2peAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoPgaXhYtWmT169fP8jdw4MCI037yySfWqVMna9KkifXs2dO2bNlS6OsLAAD8F5eZmZnp18InTpxoq1atslGjRoXGlShRwsqWLZtluu3bt1vHjh1twIAB1qZNG3v++edt48aN9te//tXi4uIizpsfZgQAoGj+MKOvvyqtAFKvXj2rXLlyntO9/vrrdsEFF9idd97pHo8ePdp+9atf2dKlS61ly5aFtLYAAMBivdlI4aV27drHnU61My1atAg9LlWqlDVq1MhWrlx5mtcQAABEG99qXtRa9e2339q//vUvmzRpkh07dsyuuuoq1+elePHiWaZNSUmx5OTkLOMqVqxoP/zwQ57LyKVFKTB27vzBDhzY7/dqRIXExNJWpUpVv1cDlMsQymT0YF8Ze2XSt/CifiwHDx50QeXZZ5+1rVu32qOPPmqHDh2y4cOHZ5nWmy6cHqenp+c6/6SkREtICO7FVLt377YBA/pYRkaG36sSFeLj423u3LlWvnx5v1clplEu/4cyGR0ok7FZJn0LL9WqVbMlS5ZYuXLlXKfbhg0bugP1Qw89ZEOGDLGEhIQsnXizBxU9zt6xN1xa2oGA17wk2HPPTfK15mXr1i02fvw4GzjwfqtevYb5fUZx9GiCpabu83U9QLn0UCajBWWyqJXJSpWivMNu9nR47rnn2uHDh23Pnj2WlJQUGl+lShVLTU3NMq0eK/Dkxb/rqApGcrK/1X/e9qtWrYadc05d81vQP8+ignL5P5TJ6ECZjL0y6Vu7yscff+yuFFKTkOfrr792gSY8uIju7bJs2bLQY71mzZo1bjwAAIgtvoWXpk2buuYg9W/ZtGmTLV682J588knr1auX67yrTrpeU9GNN95oy5cvt8mTJ9v69etds1L16tW5TBoAgBjkW3gpXbq0TZ061dLS0lw4GTZsmN18880uvOzYscNat25tK1ascNMqqDz33HP25ptvWteuXV0HLd2oLrcb1AEAgKLL1z4v5513nr388ss5xiusrFu3Lsu4tm3buj8AABDbgnstMQAAiEmEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECjFLEr07t3bkpKS7Iknnoj4/LXXXmvr1q3LMm7BggVWr169QlpDAAAQDaIivLzzzju2ePFiu+GGGyI+f+zYMfvuu+9s5syZVrt27dD4ChUqFOJaAgCAaOB7eNm9e7c9+eST1rhx41yn2bp1qx05csQuvPBCK1GiRKGuHwAAiC6+h5cxY8bYddddZ7t27cp1mg0bNthZZ51FcAEAAP6Gl08//dQ+//xz13dl5MiRuU63ceNGO+OMM6xPnz62evVqO+ecc2zQoEGuJiYvcXGnYaVjiLf9NGRbIlpQLhFtKJMxFF4OHz5sI0aMsEceecRKliyZ57Tffvut7dmzx7p162YDBw60OXPm2O23324LFy50NTKRJCUlWkICF1OdirS0RDcsXz7RKlUqc0rzAgoK5RLRhjIZQ+FlwoQJdsEFF1ibNm2OO+2oUaPs0KFDVrp0afdYtTTLly+3t956y/r27RvxNWlpB6gtOEW7dx8IDVNT953q7IACQblEtKFMFqz8nCwX8/MKo9TUVGvatKl7nJ6e7obvv/++rVixIsu0xYoVCwUXiYuLszp16tjOnTvzXEZm5mlZ9ZjhbT8N2ZaIFpRLRBvKZOHzLbzMmDHDjh49Gnr81FNPueGDDz6YY9oePXpYy5YtrX///u5xRkaGu+fLbbfdVohrDAAAYjq8VKtWLcvjxMT/9q+oVauWu69LWlqalStXzooXL27t2rWz559/3ho2bOg6606fPt327duX631hAABA0eX7pdKR7Nixwy6//HIXUlTjcscdd7gOvo8++qhramrSpIm9/PLLWZqSAABAbIia8BL+swDVq1fP8lMA6uOijrm5dc4FAACxg2uJAQBAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABAoBBeAABA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HjBdunSx0qVL26RJk2zEiBE2Z84cq1+/vk2ePJkb1J0mCii6HFpXFe3Ysd3OOuts11REjQsAIKbDi+6Yq7/cKMCEu/DCC23evHmFsGYQBZVevbgcGgAQfbiWGAAABArhBQAABArhBQAABArhBQAABArhBQAABArhBQAABArhBQAABArhBQAABArhBQAABIrvvyqN6KSfbli7do399FOaVaiQZA0anG8JCQl+rxYAAIQX5LRkySc2bdpUS0nZFRpXuXKy3X77Xday5S/ZZAAAX9FshBzB5emnn7CaNWvbY4+NtenT57ihHmu8ngcAwE+EF2RpKlKNS7NmF9mgQcOsXr0GVqpUKTfUY42fPv0vbjoAAPxCnxeEqI+Lmoruu+8hi4/Pmmv1+IYbutnw4Q+56Ro1asyWAxAVdu78wQ4c2O/b8rdt25Jl6JfExNJWpUpViwWEF4Soc67UqFEr4lapWbNmlukAwG979+6xAQP6WGZmht+rYuPHj/N1+fHx8fbSS9OtbNlyVtQRXhCiq4pky5bNrqkou++//z7LdADgNx2on3tukq81L9EiMbF0TAQXIbwgRJdD66qiuXNfd31cwpuOMjIybN681y05uYqbDgCiRaw0leB/6LCLEN3HRZdDL1/+mT355GO2bt1aO3jwZzfUY43v2fNO7vcCAPAVNS/IQvdxeeCBwe6qI3XO9ajGReO5zwsAwG+EF+SggNKiRUvusAsAiEqEF+TahMTl0ACAaESfFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAECiEFwAAEChREV7S09OtU6dOtmTJklynueeee6x+/fpZ/j788MNCXU8AAOA/338e4PDhw/bAAw/Y+vXr85xu48aNNnbsWGvVqlVoXLly5QphDQEAQDTxNbxs2LDBBZfMzMzj1sxs3brVGjdubJUrVy609QMAANHH12ajpUuXWsuWLe21117Lc7pNmzZZXFyc1ahRo9DWDQAARCdfa15uvfXWfE2n8FK6dGkbNGiQCzxVq1a1AQMGWNu2bfN8XVxcAa0ogJCdO3+wAwf2+7ZFtm3bEhr6+R1PTCxtVapU9W8FgBjme5+X/IaXQ4cOWevWra137962aNEi14FXNTZqSookKSnREhKioj8yUGTs3r3bBgzoYxkZGX6vio0fP87X5cfHx9vcuXOtfPnyvq4HEIviMo/X4aSQ6Oqh6dOnu2ak7LSj3LdvX5YOun379nX9X0aNGhVxfikp+6h5AYpgzUu0oOYFOD0qVSpTNGpedIaT/cqiOnXquA6/eYmOWAYULcnJNJV42McA/ghEu8rgwYNtyJAhWcatXbvWBRgAABBboja8pKSkuH4u0q5dO1uwYIHNnz/fNm/ebBMmTLBly5ZZ9+7d/V5NAABQyKI2vKhz7sKFC93/7du3txEjRtjEiRPdnXj/8Y9/2JQpU6x69ep+ryYAAIjVDrsFTR12AQBAsFSuXCa4NS8AAACREF4AAECgEF4AAECgEF4AAECgEF4AAECgEF4AAECgEF4AAECgEF4AAECgEF4AAECgBOJXpVH4jh07ZmvXrrGffkqzChWSrEGD8y0hIYGPAgDgO8ILcliy5BObNm2qpaTsCo2rXDnZbr/9LmvZ8pdsMQCAr/htI+QILk8//YQ1a3aRdenSzWrUqGVbtmy2uXNft+XLP7MHHhhMgAEA+PrbRoQXZGkqGjCgt9WsWdsGDRpm8fH/6xKVkZFhTz75mAsy48dPogkJAHBa8MOMOCHq46KmItW4hAcX0eMbbuhmu3btdNMBAOAXrjZCiDrnipqKIqlZs2aW6QAA8APhBSG6qkjUNBTJ999/n2U6AAD8QHhBiC6H1lVF6pyrPi7h9HjevNctObmKmw4AAL8QXhCi+7jocmhdVaTOuevWrbWDB392Qz3W+J4976SzLgDAV1xthHzd50U1Lgou3OcFAHA6cak0Thp32AUA+IHwAgAAAoX7vAAAgCKHDrsAACBQCC8AACBQCC8AACBQCC8AACBQCC8AACBQCC8AACBQCC8AACBQCC8AACBQoiK8pKenW6dOnWzJkiW5TrNmzRrr1q2bNWnSxG688UZbvXp1oa5jrNmz5yfr1+8u6969mxvqMQAgqyNH0u3tt9+yqVNfdEM9Rgz8MOPhw4ftgQcesEWLFtn06dOtZcuWOab5+eefrX379ta5c2fr2rWrzZ4929599133mjPPPDPifFNS9hXC2hdNt99+s9vm2WlbT5v2mi/rBADRZsaMl+3tt+dbRkZGaFx8fLx16nS99ejxW1/XLcii/ucBNmzYYDfddJN9//33eU63cOFCK1GihA0aNMjOPfdcGzZsmCUmJtp7771XaOsai8GlevWa9vDDf3BD0Xg9DwCxTsHlr3+da2XKlLU+ffrb5MnT3VCPNV7P4/TxNbwsXbrU1bS89lreZ/OrVq2y5s2bW1xcnHusYbNmzWzlypWFtKaxQU1DXnCZOnWWPfPM89aixcVuqMei52lCAhDL/ttUNN/KlStvL774sl1xRQerUKGCG+qxxtOEdHoVMx/deuut+ZouJSXF6tatm2VcxYoVbf369Xm+7v9nHeTTkCEPumGNGjWtXLmyWZ7T42rVati2bVvcdBMnTmW7AohJ77//rmsq+s1vutsZZ2Q9jOrxLbfcZpMmPe+m69z5Ot/WsyjzNbzk18GDB6148eJZxumxOvrmJikp0RISoqI/cmDs27fXDe+5p69VqpSzzbFPn7vtkUcecdNFeh4AYsHevT+64ZVXXmYVK+bcF2q8woumY18Zw+FF/V2yBxU9LlmyZK6vSUs7QM3LCVJb7aFDh2zixBetXr3GOZ6fNOml0HSpqXSIBhCbypat6IaLFn3omoqy03hvOvaVJy4/gS8Q4aVKlSqWmpqaZZweJycn5/k6f6+jCp7Ro5+yXr162pYt39uePXutbNn/NR3t3bvXNRl507FtAcSqDh2uthkz/mKzZ8+0tm0vt2LF/ncoPXr0qL366iyLj09w07GvPD0C0a6ie7usWLHCvKu6NVy+fLkbj4JTrlyF0KXnd911m913Xz9bsuQTN9Rj0fOaDgBi1RlnFHeXQ+/Zs9v69v2tLVr0nqWl/eiGeqzxnTpd56ZDEb3Pi6d+/fpZ7vOiTrplypRxTUP79++3K6+80jp27Gi33HKLvfrqq+4y6b/97W/c5+U04D4vAHCy93lJcMGF+7yc3vu8RG140ePRo0dbly5d3OMvvvjCRowYYRs3bnTP/fGPf7Tzzz8/1/lxk7pTo8uhdVWRmovUfKSmImpcACDnZdO6qmjnzh1WpcpZrqmIGpcYCi8FjfACAEDwRP0ddgEAAE4U4QUAAAQK4QUAAAQK4QUAAAQK4QUAAAQK4QUAAAQK4QUAAAQK4QUAAAQK4QUAAAQK4QUAAARKkf15AAAAUDRR8wIAAAKF8AIAAAKF8AIAAAKF8AIAAAKF8FKEtWvXzubOnZtjvMbpuUiee+4569GjRyGsHYqKI0eOuHJz+eWX2wUXXGC//vWvbfTo0bZ///7jvnbr1q1Wv359Nywomt+SJUsKbH6IHdovqvx4fw0aNLCLL77Y7rnnHtuxY0eh7qeRt2LHeR4A8vTUU0/ZJ598Yo8++qjVqFHDtmzZYo899pht3rzZXnzxRbYeAmXo0KF2zTXXuP8zMjJsw4YNNmLECHv44Ydt+vTpfq8e/j/CC4BTMm/ePHv88cetVatW7nH16tVt5MiRdtttt9muXbssOTmZLYzAKFOmjFWuXDn0uEqVKjZw4EB76KGHbN++fe55+I9moxjmVdk///zzdtFFF9mf/vSnUDPAsGHDrEmTJnbFFVfYwoULQ69RU8CQIUPcgUpNBFdddZX9/e9/Dz2v+b311lvWqVMn9/ytt97qzsRRdMXFxdl//vMfd5bqadq0qb3zzjtWoUKFHNXiatJROQn33nvv2aWXXmrNmjWzRx55xNLT03Nt4lSzppqpZPDgwe7v2muvdWXyu+++c+M/++wza9++vSvDv/vd72zPnj2h13/wwQd2/fXXW+PGja1FixZ2//3324EDB9xzmu8DDzzgzrS1LprnSy+9dFq2G4KjePHibhgfH5+jWTK8jGq8/lf5ad68uU2ePNmVZTWjtmnTxho1auSef+2113x7L0UF4QW2fPlye/PNN61nz55ua6xYsSL0pfzNb35jDz74oGsCEDUHfPvtt/aXv/zF3n77bbfzV9DxDjbeAUDj9PqffvrJnn32WbZyEaZyM2PGjNBO+/3337dDhw5Z3bp17YwzzsjXPObMmWPPPPOMa2b66KOPbNKkSflevsLyfffd515Tu3ZtN27WrFmuDGqo8qqDh3z//fcuzChUv/vuu65sqslLy/do/UuUKOFqlO666y7XLKZ5IDapzCiEKHwkJiYed/pt27a5/aH2fzqJ02v/+c9/uv2iQrqC86hRoyw1NbVQ1r+oIrzAbr/9dqtZs2Zox69qflX7n3vuuW7nrTOI119/3T3n1dA0bNjQTX/nnXfa7t277ccffwxtyd/+9rfujLVevXou/KxevZqtXITde++9NnbsWKtataoLAapi145egfhE+hmonKlzpMLFq6++mu/XqgZFwenCCy8Mjevfv7+1bdvW1f4NHz7cFixY4GoNVTukxzfddJNr3mrdurX98pe/tPXr14deW758ede/oVatWtarVy/3mDIcOxTAVXOoP5UthQ3tC1XG80vlRuXn7LPPdp1+ddL3i1/8wvUJ69u3r6vd9moJcXLo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",
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
        "The table and figure present the distribution of AUB across areas of residence. Respondents residing in urban areas exhibited the highest mean AUB score (M = 3.71, SD = 0.70), followed by those residing in suburban areas (M = 3.64, SD = 0.69) and rural areas (M = 3.62, SD = 0.72). The median AUB score was 4.00 for urban respondents and 3.75 for both suburban and rural respondents."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "kSis3lbwPHhP",
      "metadata": {
        "id": "kSis3lbwPHhP"
      },
      "source": [
        "While the descriptive statistics and boxplot suggest only minor differences in Actual Usage Behavior (AUB) across areas of residence, visual inspection alone is insufficient to determine whether these differences are statistically significant. Therefore, a one-way ANOVA was conducted to assess whether the mean AUB scores differ significantly among respondents residing in urban, suburban, and rural areas.\n",
        "\n",
        "For this analysis, the null hypothesis states that there is no significant difference in the mean AUB scores across the three areas of residence. A significance level of α = 0.05 is adopted for all inferential statistical tests conducted in this study.\n",
        "\n",
        "- $H_0: \\mu_{Urban} = \\mu_{Suburban} = \\mu_{Rural}$\n",
        "- $H_1:$ At least one group mean differs from the others."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 43,
      "id": "wlsoJ-67PRTl",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wlsoJ-67PRTl",
        "outputId": "8b5da290-47e5-4ed4-e165-1d3e0b2dc8f6"
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
        "A One-Way ANOVA was conducted to determine whether significant differences exist among the Urban, Suburban, and Rural groups. The analysis produced an F-statistic of 1.4681 and a p-value of 0.2310. Since the p-value exceeds the 0.05 significance level, the null hypothesis is not rejected. Thus, there is no statistically significant difference among the mean values of the three groups."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "Ym1ADl6GQPWM",
      "metadata": {
        "id": "Ym1ADl6GQPWM"
      },
      "source": [
        "#### 4. AUB Across Frequency\n",
        "\n",
        "This section explores how AUB varies across respondents with different frequencies of social media usage. Descriptive statistics and visualizations are used to examine potential patterns in AUB among the frequency groups."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 44,
      "id": "cQjBpYBVRGMF",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "cQjBpYBVRGMF",
        "outputId": "2ee54bfa-0442-428f-a8d5-de16d6452b01"
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
          "execution_count": 44,
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
      "id": "1e5b6e4c",
      "metadata": {},
      "source": [
        "To better understand the distribution, a graph showing the distribution is shown below."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 45,
      "id": "69d49339",
      "metadata": {},
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
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
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
        "The frequency groups are highly imbalanced, with most respondents reporting daily social media usage. In addition, the weekly, monthly, and rarely used categories contain relatively few observations. Based on the distributions shown above, it is difficult to meaningfully assess the normality assumption for these smaller groups. Therefore, the analysis focuses on descriptive comparisons of AUB across frequency groups rather than formal inferential testing."
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
        "The exploratory data analysis revealed that the perception and behavioral constructs, Perceived Usefulness (PU), Perceived Ease of Use (PEU), Familiarity with Social Commerce (FSC), Social Presence (SP), Trust in Platform (TP), and Interaction Behavior (IB), are all moderately to strongly positively correlated. This suggests that these constructs are interconnected and may naturally characterize different types of social commerce users.\n",
        "\n",
        "Building on this finding, the proposed research question seeks to determine whether respondents can be grouped into distinct user segments based on these constructs and whether these segments exhibit different levels of Actual Usage Behavior (AUB). Understanding these differences is important because actual usage behavior reflects the extent to which users actively engage with social commerce platforms, making it a meaningful indicator of platform adoption.\n",
        "\n",
        "The findings can help businesses and platform developers identify the characteristics of user segments that demonstrate high or low levels of actual usage. Such insights can support the development of targeted marketing strategies, personalized user experiences, and platform improvements aimed at increasing user engagement. From a research perspective, the study also provides a data-driven understanding of how perception- and behavior-based user profiles are associated with actual social commerce usage among Generation Z university students."
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "myenv",
      "language": "python",
      "name": "myenv"
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
      "version": "3.13.3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
