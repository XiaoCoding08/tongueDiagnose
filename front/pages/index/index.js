// pages/camera/camera.js
Page({
  data: {
    photoPath: ''
  },
  
  takePhoto() {
    const ctx = wx.createCameraContext();
    ctx.takePhoto({
      quality: 'high',
      success: (res) => {
        this.setData({
          photoPath: res.tempImagePath
        });
        wx.showToast({
          title: '拍照成功',
          icon: 'success'
        });
      },
      fail: (err) => {
        console.error('Error taking photo:', err);
        wx.showToast({
          title: '拍照失败',
          icon: 'none'
        });
      }
    });
  },
  
  cameraError(e) {
    console.error('Camera error:', e.detail);
    wx.showToast({
      title: '相机出错',
      icon: 'none'
    });
  },

  uploadPhoto() {
    const { photoPath } = this.data;
    console.log(photoPath)
    if (!photoPath) {
      wx.showToast({
        title: '请先拍照',
        icon: 'none'
      });
      return;
    }
    wx.navigateTo({
      url: '/pages/diagnosis/diagnosis?PicPath='+photoPath
    })
  }
});
