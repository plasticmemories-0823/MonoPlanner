"""User-owned conversation permissions; model messages cannot change them."""
from agent.storage.conversations import ConversationStore


class Permissions:
    def __init__(self, root):
        self.store = ConversationStore(root)
        with self.store.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS permissions '
                       '(thread TEXT PRIMARY KEY, requested TEXT NOT NULL, effective TEXT NOT NULL)')

    def get(self, thread):
        thread = self.store.thread(thread)
        with self.store.connection() as db:
            row = db.execute('SELECT requested,effective FROM permissions WHERE thread=?', (thread,)).fetchone()
        return dict(row) if row else {'requested': 'sandbox', 'effective': 'sandbox'}

    def request(self, thread, mode):
        if mode not in {'sandbox', 'ask', 'full'}:
            raise ValueError('请选择 sandbox、ask 或 full')
        thread = self.store.thread(thread)
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self.store.require_available(db, thread)
            db.execute('INSERT INTO permissions VALUES (?,?,?) ON CONFLICT(thread) DO UPDATE SET requested=excluded.requested',
                       (thread, mode, 'sandbox'))
        return self.get(thread)

    def boundary(self, thread):
        """Only the graph owner calls this between completed requests/operations."""
        thread = self.store.thread(thread)
        with self.store.connection() as db:
            db.execute('UPDATE permissions SET effective=requested WHERE thread=?', (thread,))
        return self.get(thread)
