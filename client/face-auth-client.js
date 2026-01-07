(function (global) {
  "use strict";

  function FaceAuthClient(options) {
    var opts = options || {};
    this.baseUrl = (opts.baseUrl || "").replace(/\/$/, "");
    this.video = opts.video;
    this.canvas = opts.canvas || document.createElement("canvas");
    this.fetchImpl = opts.fetch || global.fetch.bind(global);
    this.stream = null;
  }

  FaceAuthClient.prototype._requireVideo = function () {
    if (!this.video) {
      throw new Error("video element is required");
    }
  };

  FaceAuthClient.prototype.startCamera = async function () {
    this._requireVideo();
    if (this.stream) {
      return this.stream;
    }
    this.stream = await navigator.mediaDevices.getUserMedia({ video: true });
    this.video.srcObject = this.stream;
    await this.video.play();
    return this.stream;
  };

  FaceAuthClient.prototype.stopCamera = function () {
    if (!this.stream) {
      return;
    }
    this.stream.getTracks().forEach(function (track) {
      track.stop();
    });
    this.stream = null;
    if (this.video) {
      this.video.srcObject = null;
    }
  };

  FaceAuthClient.prototype.captureImage = function (quality) {
    this._requireVideo();
    var q = typeof quality === "number" ? quality : 0.9;
    var width = this.video.videoWidth || 640;
    var height = this.video.videoHeight || 480;
    this.canvas.width = width;
    this.canvas.height = height;
    var ctx = this.canvas.getContext("2d");
    ctx.drawImage(this.video, 0, 0, width, height);
    return this.canvas.toDataURL("image/jpeg", q);
  };

  FaceAuthClient.prototype._postJson = async function (path, payload) {
    var response = await this.fetchImpl(this.baseUrl + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    var data = await response.json();
    if (!response.ok) {
      var err = new Error(data.error || "Request failed");
      err.status = response.status;
      err.payload = data;
      throw err;
    }
    return data;
  };

  FaceAuthClient.prototype.detect = async function (options) {
    var opts = options || {};
    var image = opts.image || this.captureImage(opts.quality);
    return this._postJson("/detect-frame", {
      image: image,
      consent: true,
    });
  };

  FaceAuthClient.prototype.authenticate = async function (options) {
    var opts = options || {};
    var image = opts.image || this.captureImage(opts.quality);
    return this._postJson("/authenticate-frame", {
      image: image,
      consent: true,
    });
  };

  FaceAuthClient.prototype.signup = async function (name, options) {
    var opts = options || {};
    var count = typeof opts.count === "number" ? opts.count : 6;
    var intervalMs = typeof opts.intervalMs === "number" ? opts.intervalMs : 300;
    var images = [];

    for (var i = 0; i < count; i += 1) {
      images.push(this.captureImage(opts.quality));
      if (i < count - 1) {
        await new Promise(function (resolve) {
          setTimeout(resolve, intervalMs);
        });
      }
    }

    return this._postJson("/signup-frame", {
      name: name,
      images: images,
      consent: true,
    });
  };

  global.FaceAuthClient = FaceAuthClient;
})(window);
