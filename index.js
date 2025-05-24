const functions = require('firebase-functions');
const admin = require('firebase-admin');
const axios = require('axios');

admin.initializeApp();
const db = admin.firestore();

exports.prediksiDariRealtimeKeFirestore = functions.database
  .ref('/orders/{orderId}')
  .onCreate(async (snapshot, context) => {
    const orderId = context.params.orderId;

    try {
      // Cek dulu apakah hasil prediksi sudah ada di Firestore
      const hasilRef = db.collection('hasil_prediksi').doc(orderId);
      const existing = await hasilRef.get();

      if (existing.exists) {
        console.log(`[SKIP] Prediksi sudah ada untuk orderId: ${orderId}`);
        return null; // skip
      }

      // Jika belum ada, panggil API Flask tanpa payload data
      const response = await axios.get(`https://fuzzy-logic-q56h.onrender.com/run_fuzzy?orderId=${orderId}`);

      console.log(`[SUCCESS] Prediksi diproses oleh Flask untuk orderId: ${orderId}`, response.data);
    } catch (error) {
      console.error(`[ERROR] Gagal memproses prediksi untuk orderId: ${orderId}`, {
        message: error.message,
        response: error.response?.data,
      });
    }
  });
