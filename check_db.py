import sqlite3, glob

for f in glob.glob("*.db"):
    con = sqlite3.connect(f)
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE name LIKE '%feedback%'")]
    print(f, names)
