_users = []  # 模擬資料庫

def get_all_users():
    return _users

def add_user(name, email):
    user = {
        "id": len(_users) + 1,
        "name": name,
        "email": email
    }
    _users.append(user)
    return user
