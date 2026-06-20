from flask import Flask, render_template, request, send_file
import io, os, joblib, base64
import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LinearRegression

from src.data_prep import coulomb_counter_soc, engineer_features, load_data


# Flask App Initialization
app = Flask(__name__, template_folder='templates', static_folder='static')


# Load Pre-trained Models
feature_cols = joblib.load("Models/feature_columns.pkl")
soc_model = joblib.load("Models/soc_model.sav")
soh_model = joblib.load("Models/soh_model.sav")
price_model = joblib.load("Models/price_model.sav")


# Utility Functions
def predict_with(model, df_row, features=None):
    if features is None:
        X = df_row.values.reshape(1, -1)
    else:
        X = df_row[features].values.reshape(1, -1)
    return float(model.predict(X)[0])

def _prep_features(df):
    try:
        _, df_ocv = load_data(base_path=".", use_aged=False)
    except Exception:
        df_ocv = pd.DataFrame({
            "SOC": np.linspace(0, 1, 1001),
            "V0": np.linspace(3.0, 4.2, 1001)
        })
    df = coulomb_counter_soc(df)
    df = engineer_features(df, df_ocv)
    return df


@app.route("/predict-single", methods=["POST"])
def predict_single():
    try:
        Time = float(request.form.get("Time", 0))
        Current = float(request.form.get("Current", 0))
        Voltage = float(request.form.get("Voltage", 3.7))
        Temperature = float(request.form.get("Temperature", 25))

        df = pd.DataFrame([{ "Time": Time, "Current": Current, "Voltage": Voltage, "Temperature": Temperature }])
        df = _prep_features(df)

        soc_pred = predict_with(soc_model, df.iloc[-1], feature_cols)
        soh_pred = predict_with(soh_model, df.iloc[-1], feature_cols)
        price_features = np.array([[soh_pred, df["R_hat"].iloc[-1], df["Temperature"].mean()]])
        price_pred = float(price_model.predict(price_features)[0])

        return render_template("results.html", soc=soc_pred*100.0, soh=soh_pred*100.0, price=price_pred, mode="single")
    except Exception as e:
        return f"Error: {e}", 400

@app.route("/predict-csv", methods=["POST"])
def predict_csv():
    try:
        file = request.files["csvfile"]
        df = pd.read_csv(file)
        df.columns = [c.strip().title() for c in df.columns]
        assert all(c in df.columns for c in ["Time","Current","Voltage","Temperature"]), "CSV must have Time, Current, Voltage, Temperature"

        df = _prep_features(df)

        soc_preds = soc_model.predict(df[feature_cols].values)
        soh_preds = soh_model.predict(df[feature_cols].values)
        price_features = np.column_stack([soh_preds, df["R_hat"].values, df["Temperature"].values])
        price_preds = price_model.predict(price_features)

        out = df[["Time","Current","Voltage","Temperature"]].copy()
        out["SOC_pred"] = soc_preds
        out["SOH_pred"] = soh_preds
        out["Price_pred"] = price_preds

        # ========= Graphs =========
        plots = {}

        # Actual vs Predicted SOC
        img = io.BytesIO()
        plt.figure(figsize=(6,3))
        plt.plot(out["Time"]/3600, soc_preds*100, 'r-', label="Predicted SoC")
        plt.plot(out["Time"]/3600, out["Voltage"], 'b.', label="Actual proxy (Voltage)")
        plt.xlabel("Time / h"); plt.ylabel("SoC / %"); plt.legend(); plt.tight_layout()
        plt.savefig(img, format="png"); plt.close(); img.seek(0)
        plots["soc_curve"] = base64.b64encode(img.getvalue()).decode()

        # Residuals
        residuals = (soc_preds - out["SOC_pred"]).values if "SOC_pred" in out else soc_preds*0
        img = io.BytesIO()
        plt.figure(figsize=(6,3))
        plt.scatter(soc_preds*100, residuals, c='g')
        plt.axhline(0,color='k',linestyle='--'); plt.xlabel("Predicted SoC"); plt.ylabel("Residuals")
        plt.title("Residuals vs Predictions"); plt.tight_layout()
        plt.savefig(img, format="png"); plt.close(); img.seek(0)
        plots["residuals"] = base64.b64encode(img.getvalue()).decode()

        # Histogram of residuals
        img = io.BytesIO()
        plt.figure(figsize=(6,3))
        plt.hist(residuals, bins=30, color='c', edgecolor='k')
        plt.title("Residual Distribution"); plt.xlabel("Residual"); plt.ylabel("Count")
        plt.tight_layout(); plt.savefig(img, format="png"); plt.close(); img.seek(0)
        plots["residual_hist"] = base64.b64encode(img.getvalue()).decode()

        # SHAP values
        explainer = shap.TreeExplainer(soc_model)
        shap_values = explainer.shap_values(df[feature_cols].values[:100])
        img = io.BytesIO()
        shap.summary_plot(shap_values, df[feature_cols].iloc[:100], show=False)
        plt.tight_layout(); plt.savefig(img, format="png"); plt.close(); img.seek(0)
        plots["shap"] = base64.b64encode(img.getvalue()).decode()

        # LIME explanation
        explainer = LimeTabularExplainer(df[feature_cols].values, feature_names=feature_cols, verbose=True, mode="regression")
        exp = explainer.explain_instance(df[feature_cols].iloc[0].values, soc_model.predict, num_features=3)
        img = io.BytesIO(); fig = exp.as_pyplot_figure()
        plt.tight_layout(); fig.savefig(img, format="png"); plt.close(fig); img.seek(0)
        plots["lime"] = base64.b64encode(img.getvalue()).decode()

        # Surrogate linear model
        surrogate = LinearRegression()
        surrogate.fit(df[feature_cols].values, soc_preds)
        img = io.BytesIO()
        plt.figure(figsize=(6,3))
        plt.bar(feature_cols, surrogate.coef_)
        plt.xticks(rotation=30); plt.title("Surrogate Model Feature Importance")
        plt.tight_layout(); plt.savefig(img, format="png"); plt.close(); img.seek(0)
        plots["surrogate"] = base64.b64encode(img.getvalue()).decode()

        # Output CSV
        out_csv = out.to_csv(index=False).encode("utf-8")

        return render_template("results.html", mode="csv", table=out.head(20).to_html(classes="table table-striped", index=False), csv_data=out_csv, plots=plots)
    except Exception as e:
        return f"Error: {e}", 400

@app.route("/download", methods=["POST"])
def download():
    try:
        csv_data = request.form["csv_data"].encode("latin1")
        return send_file(io.BytesIO(csv_data), mimetype="text/csv", as_attachment=True, download_name="predictions.csv")
    except Exception as e:
        return f"Error: {e}", 400

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    return render_template("home.html")

@app.route('/about1')
def about1():
    return render_template('about1.html')

@app.route('/about2')
def about2():
    return render_template('about2.html')

@app.route('/about3')
def about3():
    return render_template('about3.html')


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')



@app.route("/signup", methods=['GET'])
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')

    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
    con.commit()
    con.close()
    return render_template("signin.html")


@app.route("/signin",methods=['GET'])
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signin.html")

@app.route('/notebook1')
def notebook1():
    return render_template('Preprocessing.html')

@app.route('/notebook2')
def notebook2():
    return render_template('SoC.html')

@app.route('/notebook3')
def notebook3():
    return render_template('SoH.html')

@app.route('/notebook4')
def notebook4():
    return render_template('Price.html')

if __name__ == '__main__':
    app.run(debug=False)