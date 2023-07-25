import sqlite3

class Database:

    def __init__(self, db):
        self.conn = sqlite3.connect(db)
        # self.cur.execute("CREATE TABLE IF NOT EXISTS book (id INTEGER PRIMARY KEY, title TEXT, author TEXT, year INTEGER, isbn INTEGER)")
        # self.conn.commit()


def main():
    database = Database("optimization.db")

if __name__ == '__main__':
    main()