Page({
  data: {
    PicPath: '',
    result: '',
    proposeDrunk: '',
    desc: '',
    proposeDrunkPicSrc: '',
    price: ''
  },
  
  onLoad: function(options) {
    // 获取传递过来的 PicPath 参数
    let PicPath = decodeURIComponent(options.PicPath);
    this.setData({
      PicPath: PicPath
    });

    // 发送请求
    this.sendRequest(PicPath);
  },
  
  sendRequest: function(imagePath) {
    wx.uploadFile({
      filePath: imagePath,
      name: 'image',
      url: 'http://47.113.110.38:5000/detect',
      success: (res) => {
        const data = JSON.parse(res.data)
        console.log(data)
        if(data.code==200){
          const r = data.data.class + '性'
          this.setResult(r)
        }
        if(data.code==201){
          wx.showToast({
            title: '检测不到舌头',
            icon: 'error',
            duration: 1000,
          });
          setTimeout(() => {
            wx.navigateTo({
              url: '/pages/index/index',
            });
          }, 1000);
          
        }
      },
      fail: (err) => {
        this.setData({
          result: '请求失败',
        });
        console.error(err);
      },
    });
  },

  setResult: function(result) {
    this.setData({
      result: result,
    });

    switch(result) {
      case '寒性':
        this.setData({
          proposeDrunk: '暖胃饮',
          proposeDrunkPicSrc: '/images/PianHan.png',
          desc: '采用山楂、佛手，普洱茶等暖胃理气',
          price: '16'
        });
        break;
      case '热性':
        this.setData({
          proposeDrunk: '清热养胃饮',
          proposeDrunkPicSrc: '/images/PianRe.png',
          desc: '采用山药、扁豆花，沙棘汁、野蜂蜜等滋阴清热养胃',
          price: '18'
        });
        break;
      case '中性':
        this.setData({
          proposeDrunk: '平胃饮',
          proposeDrunkPicSrc: '/images/Normal.png',
          desc: '采用陈皮、茯苓，百香果等理气养胃',
          price: '17'
        });
        break;
        case '胃寒阴虚内热性':
          this.setData({
            proposeDrunk: '平胃饮',
            proposeDrunkPicSrc: '/images/Normal.png',
            desc: '采用陈皮、茯苓，百香果等理气养胃',
            price: '17'
          });
          break;
      default:
        this.setData({
          result: '未知结果',
        });
        break;
    }
  }
});
