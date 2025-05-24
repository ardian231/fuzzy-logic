const functions = require('firebase-functions');
const admin = require('firebase-admin');
const axios = require('axios');

admin.initializeApp();
const db = admin.firestore();

exports.prediksiDariRealtimeKeFirestore = functions.database
  .ref('/orders/{orderId}')
  .onCreate(async (snapshot, context) => {
    const data = snapshot.val();
    const orderId = context.params.orderId;

    try {
      // Cek apakah hasil prediksi sudah ada di Firestore
      const hasilRef = db.collection('hasil_prediksi').doc(orderId);
      const existing = await hasilRef.get();
      if (existing.exists) {
        console.log(`[SKIP] Prediksi sudah ada untuk orderId: ${orderId}`);
        return null;
      }

      // Siapkan form data
      const formData = new URLSearchParams();
      formData.append('gaji', data.income || '');
      formData.append('cicilan_lain', data.installment || '');
      formData.append('pengajuan_baru', data.nominal || '');
      formData.append('pekerjaan', data.job || '');
      formData.append('item', data.item || '');
      formData.append('usia', data.usia || '30');
      formData.append('tinggal_di_kost', 'tidak');

      // Panggil API Flask
      const response = await axios.post(
        'https://your-flask-api-url/prediksi',
        formData.toString(),
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
          }
        }
      );

      // Simpan hasil prediksi
      await hasilRef.set({
        ...response.data,
        original_order: data,
        timestamp: admin.firestore.FieldValue.serverTimestamp()
      });

      console.log(`[SUCCESS] Prediksi disimpan di Firestore untuk orderId: ${orderId}`);
    } catch (error) {
      console.error(`[ERROR] Gagal memproses prediksi untuk orderId: ${orderId}`, {
        message: error.message,
        response: error.response?.data,
      });
    }
  });
