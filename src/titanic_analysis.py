print("Titanic Data Analysis Project")
print("Project setup completed successfully!")
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
##import data
data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

data_path = Path("data") / "titanic.csv"
data_path.parent.mkdir(exist_ok=True)
if data_path.exists():
    print("Using local Titanic CSV.")
    df = pd.read_csv(data_path)
else:
    print("Downloading Titanic CSV.")
    df = pd.read_csv(data_url)
    df.to_csv(data_path, index=False)
    print(f"CSV saved to: {data_path}")
#see the data
print("\n First five rows")
print(df.head)
print("\n dataset shape")
print(df.shape)
print("\n column names")
print(df.columns.tolist())

#see the basic info
df.info()
missing_values=df.isna().sum().sort_values(ascending=False)
print(missing_values)
print("missing values")
print(missing_values[missing_values>0])

#caculate the surival number and rate
survial_counts=df["Survived"].value_counts().sort_index()
print("\n survial counts:")
print(survial_counts)
survial_rate=df["Survived"].mean()
print(f"oveall survial rate: {survial_rate:.2%}")

#compare male and female survial rates
survial_by_sex=df.groupby("Sex")["Survived"].mean()
print(f"\n survival rate  by sex :{survial_by_sex}")

survial_by_class=df.groupby("Pclass")["Survived"].mean()
print("\nSURVIVAL RATE BY CLASS")
print(survial_by_class)
#==========================================================================
images_path=Path("images")
images_path.mkdir(exist_ok=True)
survial_counts.plot(kind="bar")
#make the plot
survial_counts.plot(kind="bar")

plt.title("Titanic Survival Counts")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")

plt.xticks (
    ticks=[0,1],
    labels=["Did Not Survive", "Survived"],
    rotation=0

)
plt.tight_layout()
plt.savefig(
    images_path /"survial_counts.png",
    dpi=300
)
plt.show()