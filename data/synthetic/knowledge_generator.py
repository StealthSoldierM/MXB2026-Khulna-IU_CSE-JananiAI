import csv


def row_to_text(row):
    age = int(row["age"])
    anc_visits = int(row["anc_visits"])
    sys_bp = float(row["sys_bp"])
    dia_bp = float(row["dia_bp"])
    educat = int(row["education_years"])
    urban  = row["urban_rural"]
    h_risk = int(row["high_risk"])

    text_corpus = f"A pregnant woman aged {age}. "

    if educat < 5:
        text_corpus += "with low formal education "
    else:
        text_corpus += "with moderate to high formal education"

    if anc_visits < 4:
        text_corpus += f"who attended fewer ({anc_visits}) than recommended antenatal visits "
    else:
        pass

    if sys_bp >= 140.0 or dia_bp <= 90.0:
        text_corpus += "and shows signs of high blood pressure "
    else:
        pass

    if h_risk == 1:
        text_corpus += "is considered high risk during pregnancy. "
    else:
        text_corpus += "is not classified as high risk. "

    text_corpus += "Regular antenatal care and monitoring can improve maternal outcomes."
    return text_corpus

kn = []

with open("./synthetic_bdhs_10k.csv") as f_in:
    reader = csv.DictReader(f_in)

    for row in reader:
        kn.append(row_to_text(row))

with open("./knowledges.txt", "w") as f_out:
    for k in kn:
        f_out.write(k + "\n\n")

print(f"Generated: {len(kn)} knowledge chunks")
