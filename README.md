# Menjalankan Program
## 1 - Instal pip
Melakukan penginstalan di Arch Linux dengan perintah:
```
sudo pacman -S python-pip
```
# 2 - Membuat Virtual Environment
Sebelum membuat-nya pastikan sudah diinstal, di Arch Linux dengan perintah:
```
sudo pacman -S python-virtualenv
```

Membuat Virtual Environment, dengan perintah:
```
python -m venv myenv
```

Karena saya menggunakan shell/teriminal fish, aktifkan dengan perintah:
```
source myenv/bin/activate.fish
```

Ketika ingin menonaktikan-nya dengan perintah:
```
deactivate
```
# 3 - Instal Library Python
Gunakan perintah:
```
pip install [nama library]
```
# 4 - Menjalankan File Python
Gunakan perintah di terminal:
```
python [nama file].py
```

```
python3 analisis_log.py --out-dir figures

# Atau

python3 analisis_log.py --time-mode global --out-dir figures
```


# Referensi:
https://chat.deepseek.com/a/chat/s/bcabe187-7d54-49f7-aa38-8186b97eb59b diakses menggunakan akun google pribadi.

https://g.co/gemini/share/2c1bf16e514f diakses menggunakan akun google pribadi.