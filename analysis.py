"""SAS821S Lab 2 - completed analysis.
Omatako Financial Services (OFS) - The Silent Ledger
Run from the folder containing the 02_Data CSV files.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                              recall_score, f1_score, ConfusionMatrixDisplay)

RANDOM_SEED = 42
DATA = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 0. LOAD
# ---------------------------------------------------------------------------
employees = pd.read_csv(DATA / "ofs_employee_directory.csv")
assets = pd.read_csv(DATA / "ofs_asset_inventory.csv")
policy = pd.read_csv(DATA / "ofs_access_policy_matrix.csv")
email = pd.read_csv(DATA / "ofs_email_gateway_logs.csv", parse_dates=["timestamp"])
vpn = pd.read_csv(DATA / "ofs_vpn_authentication_logs.csv", parse_dates=["timestamp"])
endpoint = pd.read_csv(DATA / "ofs_endpoint_process_logs.csv", parse_dates=["timestamp"])
files = pd.read_csv(DATA / "ofs_file_access_logs.csv", parse_dates=["timestamp"])
proxy = pd.read_csv(DATA / "ofs_proxy_dlp_logs.csv", parse_dates=["timestamp"])
intel = pd.read_csv(DATA / "ofs_threat_intelligence.csv", parse_dates=["first_seen", "last_updated"])
training = pd.read_csv(DATA / "ofs_access_session_training.csv", parse_dates=["session_start"])
investigation = pd.read_csv(DATA / "ofs_access_session_investigation.csv", parse_dates=["session_start"])

print("Row counts")
for name, frame in {
    "employees": employees, "assets": assets, "policy": policy,
    "email": email, "vpn": vpn, "endpoint": endpoint, "files": files,
    "proxy": proxy, "intel": intel, "training": training,
    "investigation": investigation,
}.items():
    print(f"{name:15s} rows={len(frame):5d}  dupes={frame.duplicated().sum():4d}  "
          f"missing={int(frame.isna().sum().sum()):4d}")
    print(frame.dtypes.to_string())
    print("-" * 60)

# ---------------------------------------------------------------------------
# TODO 1 (done): hour / weekend / off-hours derived fields
# ---------------------------------------------------------------------------
emp_hours = employees.set_index("user_id")[["normal_start", "normal_end", "department"]]

def add_time_features(df, ts_col="timestamp", user_col="user_id"):
    df = df.copy()
    df["hour"] = df[ts_col].dt.hour
    df["weekday"] = df[ts_col].dt.dayofweek          # 0=Mon .. 6=Sun
    df["is_weekend"] = df["weekday"] >= 5
    df = df.merge(emp_hours, left_on=user_col, right_index=True, how="left")
    start_h = df["normal_start"].str.split(":").str[0].astype(float)
    end_h = df["normal_end"].str.split(":").str[0].astype(float)
    df["off_hours"] = ~df["hour"].between(start_h, end_h)
    df["off_hours"] = df["off_hours"].fillna(True)
    return df

vpn = add_time_features(vpn)
files = add_time_features(files)
proxy = add_time_features(proxy)
endpoint = add_time_features(endpoint)

# new geography / new device per user (first time an IP-country or device_id is seen)
vpn = vpn.sort_values("timestamp")
vpn["seen_country_before"] = vpn.groupby("user_id")["country"].transform(
    lambda s: s.shift().eq(s).cummax().fillna(False) | s.duplicated())
vpn["new_geo"] = ~vpn.groupby("user_id")["country"].apply(
    lambda s: s.duplicated()).reset_index(level=0, drop=True)
vpn["new_device"] = ~vpn.groupby("user_id")["device_id"].apply(
    lambda s: s.duplicated()).reset_index(level=0, drop=True)

# download / upload MB
files["download_mb"] = np.where(files["action"].isin(["DOWNLOAD", "READ"]),
                                 files["bytes_transferred"] / (1024 ** 2), 0.0)
proxy["upload_mb"] = proxy["bytes_out"] / (1024 ** 2)

# resource sensitivity + role mismatch (compare accessed resource_group with the
# user's permitted_resource_groups from the employee directory)
perm = employees.set_index("user_id")["permitted_resource_groups"].str.split(";")
files = files.merge(perm.rename("permitted"), left_on="user_id", right_index=True, how="left")
files["role_mismatch"] = files.apply(
    lambda r: isinstance(r["permitted"], list) and r["resource_group"] not in r["permitted"], axis=1)
files["is_sensitive"] = files["sensitivity"].isin(["Confidential", "Restricted"])

# ---------------------------------------------------------------------------
# TODO 2 (done): individual + peer-group (department) baselines
# ---------------------------------------------------------------------------
files_dept = files  # 'department' already attached by add_time_features()
vpn_dept = vpn      # 'department' already attached by add_time_features()

user_baseline = files_dept.groupby("user_id").agg(
    avg_download_mb=("download_mb", "mean"),
    sensitive_access_rate=("is_sensitive", "mean"),
    role_mismatch_count=("role_mismatch", "sum"),
).round(2)

peer_baseline = files_dept.groupby("department").agg(
    dept_avg_download_mb=("download_mb", "mean"),
    dept_sensitive_rate=("is_sensitive", "mean"),
    dept_role_mismatch_rate=("role_mismatch", "mean"),
).round(2)

vpn_peer = vpn_dept.groupby("department").agg(
    dept_failed_rate=("result", lambda s: (s == "FAILURE").mean()),
    dept_offhours_rate=("off_hours", "mean"),
).round(3)

baseline_summary = user_baseline.merge(
    employees.set_index("user_id")["department"], left_index=True, right_index=True
).reset_index().rename(columns={"index": "user_id"})
baseline_summary = baseline_summary.merge(peer_baseline, on="department").merge(
    vpn_peer, on="department", how="left")
baseline_summary = baseline_summary.set_index("user_id")
baseline_summary.to_csv(DATA / "_baseline_summary_table.csv")   # <- summary table for B2
print(baseline_summary.sort_values("avg_download_mb", ascending=False).head(10))

# ---------------------------------------------------------------------------
# TODO 3 (done): three labelled visualisations
# ---------------------------------------------------------------------------
# Viz 1: download volume by user vs department peer average
top_users = files_dept.groupby("user_id")["download_mb"].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 5))
top_users.plot.bar(ax=ax)
ax.set_ylabel("Total downloaded MB (7-14 Sep 2026)")
ax.set_xlabel("User")
ax.set_title("Q: Which users download the most data?\nDownloaded volume by user")
plt.tight_layout()
plt.savefig(DATA / "_viz1_download_by_user.png", dpi=150)
plt.close()

# Viz 2: off-hours VPN logins by department (peer-group comparison)
offhours_rate = vpn_dept.groupby("department")["off_hours"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
offhours_rate.plot.bar(ax=ax, color="darkorange")
ax.set_ylabel("Share of VPN logins outside normal hours")
ax.set_title("Q: Which departments log in off-hours most often?\nOff-hours VPN logins by department")
plt.tight_layout()
plt.savefig(DATA / "_viz2_offhours_by_department.png", dpi=150)
plt.close()

# Viz 3: failed authentications over time for the account of interest vs peers
fail_by_hour = vpn[vpn["result"] == "FAILURE"].groupby(
    vpn["timestamp"].dt.floor("h"))["event_id"].count()
fig, ax = plt.subplots(figsize=(9, 4))
fail_by_hour.plot(ax=ax, marker="o", color="crimson")
ax.set_ylabel("Failed VPN authentications per hour")
ax.set_title("Q: When do authentication failures cluster?\nFailed VPN logins over time")
plt.tight_layout()
plt.savefig(DATA / "_viz3_failed_auth_timeline.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# TODO 3b (done): transparent rule-based risk score (B3) - >=4 indicators
# ---------------------------------------------------------------------------
sess_risk = training.copy()
sess_risk["rule_score"] = (
    3 * sess_risk["new_device"]
    + 2 * sess_risk["off_hours"]
    + 2 * sess_risk["impossible_travel"]
    + 2 * sess_risk["privilege_mismatch"]
    + 1 * (sess_risk["failed_logins_30m"] >= 2).astype(int)
    + 2 * (sess_risk["data_upload_mb"] > 50).astype(int)
    + 1 * (sess_risk["sensitive_resources"] >= 4).astype(int)
)
RULE_THRESHOLD = 8  # >= 8 of a max of 13 -> "high risk"; tune against your own results
sess_risk["rule_flag"] = sess_risk["rule_score"] >= RULE_THRESHOLD
print("\nRule-based flags vs actual label (training data):")
print(pd.crosstab(sess_risk["rule_flag"], sess_risk["label"]))

# ---------------------------------------------------------------------------
# TODO 4 (done): correlate event IDs into one incident timeline
# ---------------------------------------------------------------------------
WINDOW = ("2026-09-11 16:50:00", "2026-09-12 01:00:00")
FOCUS_USER = "mhaingura"

def slice_window(df, ts_col="timestamp"):
    return df[(df[ts_col] >= WINDOW[0]) & (df[ts_col] <= WINDOW[1])]

tl_email = slice_window(email[email["recipient"] == FOCUS_USER])[
    ["event_id", "timestamp", "subject", "url", "verdict"]].rename(columns={"subject": "detail"})
tl_email["source"] = "Email gateway"

tl_vpn = slice_window(vpn[vpn["user_id"] == FOCUS_USER])[
    ["event_id", "timestamp", "source_ip", "country", "result", "mfa_result"]]
tl_vpn["detail"] = tl_vpn["source_ip"] + " (" + tl_vpn["country"] + ") " + tl_vpn["result"] + "/" + tl_vpn["mfa_result"]
tl_vpn["source"] = "VPN"
tl_vpn = tl_vpn[["event_id", "timestamp", "detail", "source"]]

tl_ep = slice_window(endpoint[endpoint["user_id"] == FOCUS_USER])[
    ["event_id", "timestamp", "process_name", "command_line", "risk_level"]]
tl_ep["detail"] = tl_ep["process_name"] + " | " + tl_ep["command_line"].astype(str) + " | " + tl_ep["risk_level"]
tl_ep["source"] = "Endpoint (EDR)"
tl_ep = tl_ep[["event_id", "timestamp", "detail", "source"]]

tl_files = slice_window(files[files["user_id"] == FOCUS_USER])[
    ["event_id", "timestamp", "resource_path", "resource_group", "action", "status", "bytes_transferred"]]
tl_files["detail"] = (tl_files["action"] + " " + tl_files["resource_group"] + " "
                       + tl_files["resource_path"] + " (" + tl_files["status"] + ", "
                       + tl_files["bytes_transferred"].astype(str) + "B)")
tl_files["source"] = "File/KYC server"
tl_files = tl_files[["event_id", "timestamp", "detail", "source"]]

tl_proxy = slice_window(proxy[proxy["user_id"] == FOCUS_USER])[
    ["event_id", "timestamp", "destination_domain", "bytes_out", "dlp_rule", "action"]]
tl_proxy["detail"] = (tl_proxy["destination_domain"] + " " + tl_proxy["bytes_out"].astype(str)
                       + "B DLP=" + tl_proxy["dlp_rule"] + " " + tl_proxy["action"])
tl_proxy["source"] = "Proxy/DLP"
tl_proxy = tl_proxy[["event_id", "timestamp", "detail", "source"]]

timeline = pd.concat([tl_email, tl_vpn, tl_ep, tl_files, tl_proxy], ignore_index=True)
timeline = timeline.sort_values("timestamp").reset_index(drop=True)
timeline.to_csv(DATA / "_incident_timeline.csv", index=False)
print(f"\nBuilt correlated timeline with {len(timeline)} events for {FOCUS_USER} "
      f"between {WINDOW[0]} and {WINDOW[1]}")
print(timeline.to_string(index=False))

# match against threat intelligence indicators
ti_ips = set(intel["indicator"])
matched_proxy = proxy[proxy["destination_ip"].isin(ti_ips) | proxy["destination_domain"].isin(ti_ips)]
matched_hash = endpoint[endpoint["sha256"].isin(ti_ips)]
print("\nProxy events matching threat intel indicators:")
print(matched_proxy[["event_id", "timestamp", "user_id", "destination_domain", "destination_ip"]])
print("\nEndpoint events matching threat intel file hash indicators:")
print(matched_hash[["event_id", "timestamp", "user_id", "process_name", "sha256"]])

# ---------------------------------------------------------------------------
# TODO 5 (done): stratified 70/30 split + classifier
# ---------------------------------------------------------------------------
FEATURES = ["new_device", "off_hours", "failed_logins_30m", "mfa_denials",
            "unique_resources", "sensitive_resources", "denied_accesses",
            "files_read", "bytes_downloaded_mb", "data_upload_mb",
            "process_risk_score", "destination_risk_score", "peer_deviation_score",
            "impossible_travel", "privilege_mismatch"]
X = training[FEATURES]
y = training["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y)

clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_SEED,
                              class_weight="balanced")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# TODO 5b (done): evaluation metrics + confusion matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
fnr = fn / (fn + tp)

print("\nConfusion matrix [[TN FP][FN TP]]:")
print(cm)
print(f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  "
      f"F1={f1:.3f}  FalseNegativeRate={fnr:.3f}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Suspicious"])
disp.plot(cmap="Blues")
plt.title("Confusion matrix - access-session classifier")
plt.tight_layout()
plt.savefig(DATA / "_viz4_confusion_matrix.png", dpi=150)
plt.close()

feat_importance = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importance:")
print(feat_importance)

# ---------------------------------------------------------------------------
# TODO 6 (done): score investigation sessions, export top 15
# ---------------------------------------------------------------------------
inv_X = investigation[FEATURES]
investigation["risk_probability"] = clf.predict_proba(inv_X)[:, 1]
investigation["model_flag"] = investigation["risk_probability"] >= 0.5


def interpret(row):
    reasons = []
    if row["impossible_travel"]:
        reasons.append("impossible travel")
    if row["privilege_mismatch"]:
        reasons.append("privilege mismatch")
    if row["data_upload_mb"] > 50:
        reasons.append(f"large upload ({row['data_upload_mb']:.1f} MB)")
    if row["new_device"]:
        reasons.append("new device")
    if row["off_hours"]:
        reasons.append("off-hours")
    return "; ".join(reasons) if reasons else "elevated behavioural score"


investigation["interpretation"] = investigation.apply(interpret, axis=1)
top15 = investigation.sort_values("risk_probability", ascending=False).head(15)
top15_out = top15[["session_id", "session_start", "user_id", "risk_probability", "interpretation"]]
top15_out.to_csv(DATA / "top_15_high_risk_sessions.csv", index=False)
print("\nTop 15 highest-risk investigation sessions:")
print(top15_out.to_string(index=False))

# Compare model ranking with the rule-based score computed the same way on
# the investigation set (same rule as above, adapted to available columns)
investigation["rule_score"] = (
    3 * investigation["new_device"]
    + 2 * investigation["off_hours"]
    + 2 * investigation["impossible_travel"]
    + 2 * investigation["privilege_mismatch"]
    + 1 * (investigation["failed_logins_30m"] >= 2).astype(int)
    + 2 * (investigation["data_upload_mb"] > 50).astype(int)
    + 1 * (investigation["sensitive_resources"] >= 4).astype(int)
)
compare = investigation[["session_id", "risk_probability", "rule_score"]].sort_values(
    "risk_probability", ascending=False).head(15)
print("\nModel probability vs rule score for the top-ranked sessions:")
print(compare.to_string(index=False))

print("\nDone. Outputs written next to the data: "
      "_baseline_summary_table.csv, _viz1..4 PNGs, _incident_timeline.csv, "
      "top_15_high_risk_sessions.csv")
