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
survival_counts=df["Survived"].value_counts().sort_index()
print("\n survial counts:")
print(survival_counts)
survial_rate=df["Survived"].mean()
print(f"oveall survial rate: {survial_rate:.2%}")

#compare male and female survial rates
survival_by_sex=df.groupby("Sex")["Survived"].mean()
print(f"\n survival rate  by sex :{survival_by_sex}")

survival_by_class=df.groupby("Pclass")["Survived"].mean()
print("\nSURVIVAL RATE BY CLASS")
print(survival_by_class)
#==========================================================================
images_path=Path("images")
images_path.mkdir(exist_ok=True)
#make the plot
survival_counts.plot(kind="bar")

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
    images_path /"survival_counts.png",
    dpi=300
)
plt.show()





ax=survival_by_sex.plot(kind="bar")
plt.title("Titanic Survival Rate By sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.xticks(rotation=0)
plt.ylim(0,1)
labels=[f"{value:.1%}"for value in survival_by_sex.values]
ax.bar_label(
    ax.containers[0],
    labels=labels,
    padding=3
)
plt.tight_layout()
plt.savefig(
    images_path/"survival_rate_by_sex.png",
    dpi=300
)
plt.show()
# make the Pclass plot
print(survival_by_class)
ax=survival_by_class.plot(kind="bar")
plt.title("Titanic survival Rate by Passanger Class")
plt.xlabel("Passanger Class")
plt.ylabel("Survival Rate")
plt.ylim(0,1)

labels=[f"{value:.1%}" for value in survival_by_class.values]
ax.bar_label(
    ax.containers[0],
    labels=labels,
    padding=3
)
plt.xticks(
    ticks=[0,1,2],
    labels=["1st Class","2nd Class","3rd Class"],
    rotation=0
)
plt.show()
plt.savefig(
    images_path/"survival_rate_by_Pclass.png",
    dpi=300)
#two factors
survival_by_class_sex=(
    df.groupby(["Sex","Pclass"])["Survived"]
    .mean()
)
print(survival_by_class_sex)
#------

survival_table = survival_by_class_sex.unstack()
print("\nSURVIVAL RATE BY SEX AND CLASS")
print(survival_table)
ax=survival_table.plot(kind="bar")
plt.title("Titanic Survival Rate by Sex and Passenger Class")
plt.xlabel("Sex")
plt.ylabel("Survial Rate")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.legend(
    title="Passenger Class",
    labels=["1st Class", "2nd Class", "3rd Class"]
)
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f",
        padding=3
    )


plt.savefig(
    images_path/"Survival Rate by sex and pclass",
    dpi=300
)
plt.show()
