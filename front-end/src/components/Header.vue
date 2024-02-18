<template>
  <div id="Header">
    <div id="word">
      <h1>{{ msg }}</h1>
    </div>
  </div>
</template>
<script>
export default {
  name: "Header",
  data() {
    return {
      msg: "CNN-Transformer-WEB端",
      activeIndex: "1",
    };
  },
  methods: {},
};
</script>
<style scoped>
#Header {
  display: flex;
  justify-content: center; /* 水平居中 */
  align-items: center; /* 垂直居中 */
  padding: 30px 110px 0 150px;
  width: 90%;
  margin: 10px auto;
  height: 100px; /* 指定一个高度，如果需要的话 */
}

#word {
  margin-left: -11%;
  margin-top: -35px;
  margin-bottom: 37px;
  height: 60px;
  line-height: 3.2em;
  letter-spacing: 8px;
}

h1 {
  /*text-align: center;*/
  color: #21b3b9;
  letter-spacing: 30px;
  font-size: 2.3em;
}

.el-menu-demo {
  width: 80%;
  margin: 0px auto;
  padding: 0px auto;
}

.top-left-edition span i {
  float: left;
  margin-right: 10px;
}

i,
input,
label {
  vertical-align: middle;
}

i {
  border: 0;
  display: block;
  cursor: pointer;
}

.top-left-edition span {
  float: left;
  font-size: 16px;
  color: #999999;
  line-height: 24px;
  margin-right: 40px;
}
</style>


