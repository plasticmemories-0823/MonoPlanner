"""Local display-only stream; fragments never authorize work or restore cursors."""
import json
import time
from agent.storage.records import save


class LiveReasoning:
    def __init__(self, root, call, after_id):
        self.path = root/'display/model.json'
        self.fragments = root/'requests'/f'call-{call:04d}'/'reasoning.jsonl'
        self.row = {'id': f'{root.name}-thinking-{call}', 'task_id': root.name,
                    'after_id': after_id, 'kind': 'thinking', 'label': '正在思考',
                    'status': 'running', 'detail': '', 'started_at': time.time(),
                    'version': call * 1_000_000}
        self.pending, self.updated = [], time.monotonic()
        save(self.path, self.row)

    def append(self, text):
        self.pending.append(text)
        if not self.row['detail'] or time.monotonic() - self.updated >= .2:
            self.flush()

    def flush(self):
        if self.pending:
            delta = ''.join(self.pending)
            self.fragments.parent.mkdir(parents=True, exist_ok=True)
            with self.fragments.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'text': delta}, ensure_ascii=False) + '\n')
            self.row['detail'] += delta
            self.pending.clear()
        self.row['version'] += 1
        save(self.path, self.row)
        self.updated = time.monotonic()

    def finish(self, status, detail=None):
        self.flush()
        self.row.update(status=status, label='思考' if status == 'completed' else '回复中断',
                        seconds=max(0, time.time() - self.row['started_at']))
        if detail is not None:
            self.row['detail'] = detail
        self.row['version'] += 1
        save(self.path, self.row)
