import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load preprocessed dataset
DATA_PATH = "world_cup_player_stats.csv"
df = pd.read_csv(DATA_PATH)

# Load all pickle bundles
with open("xg_model_bundle.pkl", "rb") as f:
  xg_bundle = pickle.load(f)

with open("position_classifier.pkl", "rb") as f:
  pos_bundle = pickle.load(f)

with open("knn_position_bundle.pkl", "rb") as f:
  knn_bundle = pickle.load(f)

# Helper to get unique sorted player names
players_list = sorted(df["player"].dropna().unique().tolist())

# Helper to get unique sorted countries/teams
countries_list = sorted(df["team"].dropna().unique().tolist()) if "team" in df.columns else []

# Helper to get unique sorted seasons (most recent first)
seasons_list = sorted(df["season_name"].dropna().unique().tolist(), reverse=True)


@app.route("/")
def home():
  show_home = request.args.get("home", "false")

  top_scorer_row = df.loc[df["goals"].idxmax()]
  top_scorer = {
      "player": top_scorer_row["player"],
      "team": top_scorer_row["team"],
      "season": int(top_scorer_row["season_name"]),
      "goals": int(top_scorer_row["goals"]),
  }

  summary = {
      "total_players": len(players_list),
      "total_seasons": len(seasons_list),
      "total_records": len(df),
      "total_goals": int(df["goals"].sum()),
      "avg_pass_completion": round(df["pass_completion_pct"].mean(), 1),
  }

  per90_labels = ["Goals", "Shots", "Key Passes", "Tackles", "Interceptions"]
  per90_values = [
      round(df["goals_p90"].mean(), 2),
      round(df["shots_p90"].mean(), 2),
      round(df["key_passes_p90"].mean(), 2),
      round(df["tackles_p90"].mean(), 2),
      round(df["interceptions_p90"].mean(), 2),
  ]

  goals_by_season = df.groupby("season_name")["goals"].sum().sort_index()
  season_labels = [str(int(s)) for s in goals_by_season.index.tolist()]
  season_values = [int(v) for v in goals_by_season.values.tolist()]

  position_counts = df["position_group"].value_counts()
  position_labels = position_counts.index.tolist()
  position_values = [int(v) for v in position_counts.values.tolist()]

  return render_template(
      "index.html",
      active_tab="home",
      show_home=show_home,
      summary=summary,
      top_scorer=top_scorer,
      per90_labels=per90_labels,
      per90_values=per90_values,
      season_labels=season_labels,
      season_values=season_values,
      position_labels=position_labels,
      position_values=position_values,
  )


@app.route("/description", methods=["GET"])
def description():
  """Player Performance Indicator Route (formerly Player Analytics)"""
  selected_player = request.args.get("player", "").strip()
  selected_season = request.args.get("season", "").strip()
  player_data = None
  player_seasons = []

  if selected_player:
    name_match = df[df["player"].str.lower() == selected_player.lower()]
    if not name_match.empty:
      selected_player = name_match.iloc[0]["player"]
      player_seasons = sorted(
          name_match["season_name"].dropna().unique().tolist(), reverse=True
      )

      if selected_season:
        try:
          season_val = type(name_match.iloc[0]["season_name"])(selected_season)
        except (TypeError, ValueError):
          season_val = selected_season
        season_match = name_match[name_match["season_name"] == season_val]
      else:
        season_match = name_match

      if not season_match.empty:
        season_match = season_match.sort_values("season_name", ascending=False)
        player_data = season_match.iloc[0].to_dict()
        selected_season = str(player_data.get("season_name", ""))

  return render_template(
      "index.html",
      active_tab="description",
      players=players_list,
      seasons=seasons_list,
      selected_player=selected_player,
      selected_season=selected_season,
      player_seasons=player_seasons,
      player_data=player_data,
  )


@app.route("/comparison", methods=["GET", "POST"])
def comparison():
  p1, p2 = None, None
  comparison_report = None
  if request.method == "POST":
    name1 = request.form.get("player1")
    name2 = request.form.get("player2")
    m1 = df[df["player"] == name1]
    m2 = df[df["player"] == name2]
    if not m1.empty:
      p1 = m1.iloc[0].to_dict()
    if not m2.empty:
      p2 = m2.iloc[0].to_dict()

    if p1 and p2:
      t1_tackles = p1.get("tackles_p90", 0)
      t2_tackles = p2.get("tackles_p90", 0)
      i1_interc = p1.get("interceptions_p90", 0)
      i2_interc = p2.get("interceptions_p90", 0)
      def_score_1 = t1_tackles + i1_interc
      def_score_2 = t2_tackles + i2_interc

      better_def = p1["player"] if def_score_1 >= def_score_2 else p2["player"]
      diff = abs(def_score_1 - def_score_2)

      comparison_report = (
          f"In defensive contributions (tackles and interceptions per 90), "
          f"{better_def} leads with a combined defensive output of {max(def_score_1, def_score_2):.2f} "
          f"per 90 minutes (outperforming by {diff:.2f}). "
          f"{p1['player']} exhibits a pass completion rate of {p1.get('pass_completion_pct', 0):.1f}% "
          f"compared to {p2['player']}'s {p2.get('pass_completion_pct', 0):.1f}%, highlighting key tactical distribution differences."
      )

  return render_template(
      "index.html",
      active_tab="comparison",
      players=players_list,
      p1=p1,
      p2=p2,
      comparison_report=comparison_report,
  )


@app.route("/replacement", methods=["GET", "POST"])
def replacement():
  selected_retiring = None
  retiring_data = None
  replacements = []
  successor_report = None

  if request.method == "POST":
    selected_retiring = request.form.get("retiring_player")
    match = df[df["player"] == selected_retiring]
    if not match.empty:
      target_row = match.iloc[0]
      retiring_data = target_row.to_dict()
      num_cols = [
          "matches_played",
          "minutes_played",
          "goals_p90",
          "shots_p90",
          "passes_p90",
          "completed_passes_p90",
          "key_passes_p90",
          "tackles_p90",
          "interceptions_p90",
          "recoveries_p90",
      ]
      valid_num_cols = [c for c in num_cols if c in df.columns]

      other_df = df[df["player"] != selected_retiring].copy()
      if not other_df.empty and valid_num_cols:
        target_vals = target_row[valid_num_cols].astype(float).values

        distances = []
        for idx, row in other_df.iterrows():
          vals = row[valid_num_cols].astype(float).values
          dist = np.linalg.norm(target_vals - vals)
          distances.append(
              (dist, row["player"], row["team"], row.get("position", "Unknown"))
          )

        distances.sort(key=lambda x: x[0])
        top_5 = distances[:5]
        max_dist = top_5[-1][0] if top_5 and top_5[-1][0] > 0 else 1.0

        for dist, p_name, p_team, p_pos in top_5:
          sim_score = max(
              0, min(100, int((1 - (dist / (max_dist * 1.5 + 1e-5))) * 100))
          )
          replacements.append({
              "player": p_name,
              "team": p_team,
              "position": p_pos,
              "similarity": f"{sim_score}%",
          })

        if replacements:
          best_match = replacements[0]
          successor_report = (
              f"To successfully mitigate the retirement of {selected_retiring} "
              f"({retiring_data.get('team', 'National Team')} - {retiring_data.get('position', 'General')}), "
              f"scouting analytics identify **{best_match['player']}** ({best_match['team']}) as the premier tactical successor "
              f"with a similarity score of **{best_match['similarity']}**. "
              f"This replacement matches the athletic workload, ball progression metrics, and defensive pressures required for seamless squad continuity."
          )

  return render_template(
      "index.html",
      active_tab="replacement",
      players=players_list,
      selected_retiring=selected_retiring,
      retiring_data=retiring_data,
      replacements=replacements,
      successor_report=successor_report,
  )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None
    form_data = None

    if request.method == 'POST':
        try:
            # Raw values entered in the form
            matches_played = float(request.form.get('matches_played', 0))
            minutes_played = float(request.form.get('minutes_played', 0))
            goals = float(request.form.get('goals', 0))
            shots = float(request.form.get('shots', 0))
            passes = float(request.form.get('passes', 0))
            completed_passes = float(request.form.get('completed_passes', 0))
            key_passes = float(request.form.get('key_passes', 0))
            tackles = float(request.form.get('tackles', 0))
            interceptions = float(request.form.get('interceptions', 0))
            recoveries = float(request.form.get('recoveries', 0))

            if minutes_played <= 0:
                raise ValueError("Minutes Played must be greater than 0.")
            if passes < 0 or completed_passes < 0 or completed_passes > passes:
                raise ValueError("Completed Passes must be between 0 and Passes.")

            form_data = request.form

            # The saved models were trained with engineered features, not the
            # 10 raw fields shown on the web form.
            xg_model = xg_bundle.get('model') if isinstance(xg_bundle, dict) else xg_bundle
            xg_scaler = xg_bundle.get('scaler') if isinstance(xg_bundle, dict) else None
            xg_columns = xg_bundle.get('feature_columns') if isinstance(xg_bundle, dict) else None

            pos_model = pos_bundle.get('model') if isinstance(pos_bundle, dict) else pos_bundle
            pos_scaler = pos_bundle.get('scaler') if isinstance(pos_bundle, dict) else None
            pos_columns = pos_bundle.get('feature_columns') if isinstance(pos_bundle, dict) else None

            if not xg_columns or not pos_columns:
                raise ValueError("Model feature_columns are missing from the pickle bundle.")

            # Start with neutral values (the scaler means) for features that
            # are not present in the current HTML form. If a player was chosen,
            # use that player's historical row for those extra features.
            player_name = request.form.get('player_name', '').strip()
            country = request.form.get('country', '').strip()

            base_row = {}

            if player_name:
                matches = df[df["player"].str.lower() == player_name.lower()]
                if country and "team" in df.columns:
                    country_matches = matches[
                        matches["team"].astype(str).str.lower() == country.lower()
                    ]
                    if not country_matches.empty:
                        matches = country_matches

                if not matches.empty:
                    base_row = matches.iloc[0].to_dict()

            def neutral_value(scaler, column):
                """Use the training mean when a feature was not supplied."""
                if scaler is not None and hasattr(scaler, "feature_names_in_"):
                    names = list(scaler.feature_names_in_)
                    if column in names:
                        return float(scaler.mean_[names.index(column)])
                return 0.0

            def make_feature_dict(columns, scaler):
                values = {
                    col: neutral_value(scaler, col)
                    for col in columns
                }
                # Use historical values where available.
                for col in columns:
                    if col in base_row and pd.notna(base_row[col]):
                        try:
                            values[col] = float(base_row[col])
                        except (TypeError, ValueError):
                            pass
                return values

            def p90(value):
                return (value / minutes_played) * 90.0

            # -------------------------
            # Build XG model input: 23 features
            # -------------------------
            xg_features = make_feature_dict(xg_columns, xg_scaler)

            # Required raw/per-90 features from the form
            raw_metrics = {
                "matches_played": matches_played,
                "minutes_played": minutes_played,
                "goals_p90": p90(goals),
                "shots_p90": p90(shots),
                "passes_p90": p90(passes),
                "completed_passes_p90": p90(completed_passes),
                "key_passes_p90": p90(key_passes),
                "tackles_p90": p90(tackles),
                "interceptions_p90": p90(interceptions),
                "recoveries_p90": p90(recoveries),

                # These are represented as ratios/flags in the trained data
                # (the UI displays pass completion as a percentage).
                "attempted_pass": 1.0 if passes > 0 else 0.0,
                "pass_completion_pct": (
                    completed_passes / passes if passes > 0 else 0.0
                ),
                "attempted_dribble": xg_features.get("attempted_dribble", 0.0),
                "dribble_success_pct": xg_features.get("dribble_success_pct", 0.0),
                "took_shot": 1.0 if shots > 0 else 0.0,
                "shot_conversion": goals / shots if shots > 0 else 0.0,
            }

            for col, value in raw_metrics.items():
                if col in xg_features:
                    xg_features[col] = value

            # season_name is a numeric feature in the XG model.
            if "season_name" in xg_features:
                if "season_name" in base_row and pd.notna(base_row["season_name"]):
                    xg_features["season_name"] = float(base_row["season_name"])
                else:
                    xg_features["season_name"] = float(
                        seasons_list[0] if seasons_list else neutral_value(xg_scaler, "season_name")
                    )

            X_xg = pd.DataFrame(
                [[xg_features[col] for col in xg_columns]],
                columns=xg_columns
            )

            # IMPORTANT: the saved model was trained after StandardScaler.
            X_xg_scaled = xg_scaler.transform(X_xg) if xg_scaler is not None else X_xg
            pred_xg = round(float(xg_model.predict(X_xg_scaled)[0]), 2)

            # -------------------------
            # Build position model input: 24 features
            # -------------------------
            pos_features = make_feature_dict(pos_columns, pos_scaler)

            for col, value in raw_metrics.items():
                if col in pos_features:
                    pos_features[col] = value

            # The classifier was trained with xg_p90 and xg_overperformance.
            # The Ridge model's output is used as the predicted xG-per-90 input.
            if "xg_p90" in pos_features:
                pos_features["xg_p90"] = pred_xg

            if "xg_overperformance" in pos_features:
                pos_features["xg_overperformance"] = p90(goals) - pred_xg

            X_pos = pd.DataFrame(
                [[pos_features[col] for col in pos_columns]],
                columns=pos_columns
            )

            X_pos_scaled = pos_scaler.transform(X_pos) if pos_scaler is not None else X_pos
            predicted_position = str(pos_model.predict(X_pos_scaled)[0])

            source_note = (
                f"Historical profile data for {base_row.get('player', player_name)} was used for "
                "additional model features."
                if base_row else
                "The entered 10 metrics were converted to the 23/24 engineered features "
                "required by the trained models."
            )

            prediction_result = {
                "xg": pred_xg,
                "position": predicted_position,
                "report": (
                    f"Based on evaluation, the model projects an Expected Goals (xG) output "
                    f"of <b>{pred_xg}</b> and classifies the tactical role as "
                    f"<b>{predicted_position}</b>.<br><br>{source_note}"
                )
            }

        except Exception as e:
            prediction_result = {
                "xg": "Error",
                "position": "N/A",
                "report": f"Prediction failed: {str(e)}"
            }

    return render_template(
        'index.html',
        active_tab='predict',
        players=players_list,
        countries=countries_list,
        prediction_result=prediction_result,
        form_data=form_data
    )

if __name__ == "__main__":
  app.run(debug=True, port=5000)