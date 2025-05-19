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
      // Siapkan data form untuk POST ke API Flask
      const formData = new URLSearchParams();
      formData.append('gaji', data.income || '');
      formData.append('cicilan_lain', data.installment || '');
      formData.append('pengajuan_baru', data.nominal || '');
      formData.append('pekerjaan', data.job || '');
      formData.append('item', data.item || '');
      formData.append('usia', data.usia || '30');
      formData.append('tinggal_di_kost', 'tidak');

      // Panggil API prediksi Flask
      const response = await axios.post('https://your-flask-api-url/prediksi', formData);

      // Simpan hasil prediksi ke Firestore, collection "hasil_prediksi"
      await db.collection('hasil_prediksi').doc(orderId).set({
        ...response.data,
        original_order: data,   // optional, simpan data asli juga
        timestamp: admin.firestore.FieldValue.serverTimestamp()
      });

      console.log(`[SUCCESS] Hasil prediksi tersimpan di Firestore untuk orderId: ${orderId}`);
    } catch (error) {
      console.error(`[ERROR] Gagal proses prediksi untuk orderId: ${orderId}`, error);
    }
  });
