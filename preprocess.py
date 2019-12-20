# coding=utf-8
from core.components.storage import Storage, load_storage, save_storage
from yan_tools.io.file import list_file_in_path

file_paths = list_file_in_path('./data/conll/')
storage = Storage.load_from_path(*file_paths, flag=3)
print('start save')
save_storage(storage, './caches/conll_storage.pkl')
