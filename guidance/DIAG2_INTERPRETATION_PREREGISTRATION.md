# Pra-registrasi interpretasi DIAG2 (proyeksi bertanda gradien)

**Ditulis 2026-10-15, SAAT run 20-pocket baru selesai 9/20.** Angka final BELUM dilihat. Tujuannya
satu: mengunci arti tiap kemungkinan hasil SEBELUM hasilnya ada, supaya interpretasi tidak bisa
dirasionalisasi setelah angkanya keluar. Ini disiplin yang sama yang dipakai Track E (Appendix D).

## 1. Apa yang diukur

Untuk tiap atom ligan di tiap state x0-hat yang ditangkap:

    cos_theta = < unit(grad_pos) , unit(atom_pocket_terdekat - posisi_atom) >

Konvensi tanda diverifikasi terhadap sampler (`guided_sampling.py` melakukan
`pos0_from_e = pos0_from_e + grad_pos`), jadi **cos > 0 = didorong MENUJU kontak pocket**. Di bawah
hipotesis nol "arah tidak berhubungan dengan pocket", E[cos] = 0.

Dua kontrol ikut dijalankan dan **harus lolos lebih dulu**, kalau tidak hasil utamanya tidak dibaca:
- **Kontrol arah acak**: vektor isotropik acak terhadap arah yang sama. Harus ≈ 0.
- **Uji beda-hingga**: melangkah sepanjang +grad harus **menaikkan** skor model sendiri. Kalau turun,
  konvensi tandanya terbalik dan semua klaim arah ikut terbalik.

Pada smoke test 1 pocket keduanya lolos (acak −0,011; beda-hingga 100% naik).

## 2. Mengapa ini penting untuk tesis

Bab V.4 saat ini melaporkan korelasi **magnitudo** gradien dengan jarak ke pocket (+0,240), dan
menyimpulkan "tarikan terkuat bekerja pada atom yang paling jauh". Tapi magnitudo adalah **norma**, dan
norma tidak membawa arah. DIAG2 mengukur arah itu langsung. Tesis sudah menyatakan pengukuran ini
terbuka; ini menutupnya.

## 3. Aturan keputusan, dikunci sekarang

Statistik utama: **mean cos_theta**, dengan CI95 bootstrap klaster-**pocket** (B=2000, 20 pocket).
Pocket adalah unit resampling, konsisten dengan seluruh tesis (Bab III.6).

| Hasil | Arti | Konsekuensi untuk naskah |
|---|---|---|
| **CI mengecualikan nol, mean NEGATIF** | Gradien rata-rata mendorong atom MENJAUHI kontak | Menguatkan narasi Bab V secara substansial. Klaim naik dari "magnitudo salah tempat" ke "arah benar-benar salah". |
| **CI mengecualikan nol, mean POSITIF** | Gradien rata-rata mendorong MENUJU kontak, meski magnitudonya terbesar di atom jauh | **MELEMAHKAN/menghaluskan narasi Bab V.** Harus ditulis sebagai koreksi: arahnya tidak salah, yang bermasalah adalah alokasi magnitudo dan/atau besarannya yang terlalu kecil. Frasa seperti "wrong direction" di Bab V, IX, X HARUS direvisi. |
| **CI memuat nol** | Arah tidak dapat dibedakan dari acak | Konsisten dengan M1 (sinyal tak informatif): gradien tidak membawa informasi arah tentang antarmuka. Dilaporkan sebagai itu, BUKAN sebagai "mendorong menjauh". |
| **Kontrol gagal** (acak jauh dari 0, atau beda-hingga turun) | Pipeline/konvensi rusak | Hasil utama TIDAK dilaporkan. Diperbaiki dulu. |

**Statistik sekunder** (dilaporkan apa pun hasilnya, tidak dipakai untuk memutuskan): fraksi atom
dengan cos > 0; mean cos pada paruh-dekat vs paruh-jauh (menguji apakah arah berubah dengan jarak);
magnitudo rata-rata.

## 4. Komitmen yang mengikat

1. **Hasil dilaporkan apa pun arahnya**, termasuk kalau ia melemahkan cerita yang sudah ditulis. Smoke
   test 1 pocket sudah memberi mean cos **+0,087** (58% menuju kontak) — artinya baris kedua tabel di
   atas adalah kemungkinan yang nyata, dan itu dicatat di sini SEBELUM hasil penuh ada supaya tidak ada
   godaan mengecilkannya nanti.
2. **Satu pocket bukan dua puluh.** Angka smoke test tidak dipakai sebagai hasil; ia hanya alasan
   mengapa kemungkinan "positif" ditulis eksplisit di sini.
3. **Tidak ada ambang yang digeser setelah melihat angka.** Aturan di tabel §3 adalah final.
4. Kalau hasilnya memaksa revisi Bab V/IX/X, revisi itu dilakukan dan **dicatat sebagai koreksi**, sama
   seperti koreksi-koreksi lain di tesis ini (mis. klaim DIAG1 soal ukuran yang dulu salah).
