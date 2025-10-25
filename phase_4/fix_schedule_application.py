# Read the file
with open('fine_tune.py', 'r') as f:
    content = f.read()

# Replace the broken schedule logic with a proper one
old_code = '''    if schedule_df is not None and "module_name" in schedule_df.columns:
        df = schedule_df.copy()
        df["predicted_rank"] = pd.to_numeric(df["predicted_rank"], errors="coerce").fillna(4).clip(0, 64)
        df = df[df["feature"].isin(cand)]
        if len(df) > 0:
            for _, row in df.iterrows():
                pr = int(row["predicted_rank"])
                tgt = str(row["module_name"])
                for th, r in r_bins:
                    if pr >= th:
                        assign[r].append(tgt)
                        break
            for r in list(assign.keys()):
                assign[r] = sorted(set(assign[r]))
            return assign'''

new_code = '''    if schedule_df is not None and "feature" in schedule_df.columns:
        df = schedule_df.copy()
        df["predicted_rank"] = pd.to_numeric(df["predicted_rank"], errors="coerce").fillna(4).clip(1, 64)
        df = df[df["feature"].isin(cand)]
        if len(df) > 0:
            # Use actual rank values, don't bucket them
            rank_map = {}  # module -> rank
            for _, row in df.iterrows():
                pr = int(row["predicted_rank"])
                tgt = str(row["feature"])
                rank_map[tgt] = pr
            
            # Group by rank value
            assign_by_rank = {}
            for tgt, r in rank_map.items():
                if r not in assign_by_rank:
                    assign_by_rank[r] = []
                assign_by_rank[r].append(tgt)
            
            # Convert to expected format: assign[rank] = [modules]
            assign = {}
            for r, mods in assign_by_rank.items():
                assign[r] = sorted(set(mods))
            
            return assign'''

content = content.replace(old_code, new_code)

# Write back
with open('fine_tune.py', 'w') as f:
    f.write(content)

print("✅ Fixed schedule application logic!")
