#### snapshot_management.py

import redis

redis_client = redis.Redis()

def save_snapshot(state):
    redis_client.set('snapshot', state)

def load_snapshot():
    return redis_client.get('snapshot')
