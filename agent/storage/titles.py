"""Conversation display metadata and auxiliary usage; never task state."""
import json
from datetime import datetime, timezone

from agent.storage.conversations import ConversationStore
from agent.storage.usage import summarize


class Titles:
    def __init__(self, root):
        self.store = ConversationStore(root)

    def get(self, thread):
        with self.store.connection() as db:
            row = db.execute('SELECT data FROM conversation_titles WHERE thread=?',
                             (self.store.thread(thread),)).fetchone()
            return json.loads(row['data']) if row else None

    def public(self, thread=None):
        with self.store.connection() as db:
            query = ('SELECT conversation_titles.* FROM conversation_titles LEFT JOIN presentation USING(thread) '
                     "WHERE (visibility IS NULL OR visibility NOT IN ('purging','purged'))")
            parameters = ()
            if thread:
                query += ' AND conversation_titles.thread=?'
                parameters = (self.store.thread(thread),)
            result = {row['thread']: self.display(json.loads(row['data'])) for row in db.execute(
                query, parameters)}
        return result.get(self.store.thread(thread)) if thread else result

    @staticmethod
    def display(record):
        return {key: record[key] for key in ('title', 'status', 'revision')}

    def update(self, thread, expected, **values):
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            thread = self.store.thread(thread)
            try:
                self.store.require_available(db, thread)
            except ValueError:
                return False
            row = db.execute('SELECT data FROM conversation_titles WHERE thread=?', (thread,)).fetchone()
            record = json.loads(row['data']) if row else None
            if not record or record['status'] not in expected:
                return False
            record.update(values, revision=record['revision'] + 1)
            db.execute('UPDATE conversation_titles SET data=? WHERE thread=?',
                       (json.dumps(record, ensure_ascii=False), thread))
            return True

    def settle_interrupted(self):
        """Startup never resends a title whose previous owner disappeared."""
        for thread in self.public():
            record = self.get(thread)
            if record and record['status'] in {'pending', 'running'}:
                self.update(thread, {record['status']},
                            status='unknown' if record.get('request_sent') else 'failed',
                            error_type='ServiceInterrupted', finished_at=datetime.now(timezone.utc).isoformat())

    def activity(self, thread):
        record = self.get(thread)
        display = self.public(thread)
        if not record or not display:
            return {'title': None, 'title_pending': False, 'usage': {}}
        rows = [{'usage': record.get('usage'), 'seconds': record.get('seconds', 0)}] if record.get('request_sent') else []
        usage = summarize({'model_usage': rows})
        usage['purpose'] = 'title'
        return {'title': display, 'title_pending': record['status'] in {'pending', 'running'},
                'usage': {'title:' + self.store.thread(thread): usage}}
