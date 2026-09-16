## ssvep 视觉刺激程序设计

原方案 qt window

实现在ui包中

类display
|-- 类VisualObj

接口
1. add_visual_obj(self, obj):
2. lambda: window.visual_objs.clear() or window.update()
3. add_visual_point(self, x, y):
4. delete_visual_point(self):

## 目标：
基于 psychopy 实现精确的每一帧的每一相位的闪烁控制
运行环境conda BCI-psychopy
实现在ui包中

类display_psy
|-- 类VisualObj_psy
接口兼容 原方案

优化点：
- 余弦波闪烁，初始相位为0
- 实时闪烁测试，每次1.5s的闪烁时间每帧的实际相位相位记录在record/{file_name}里，程序运行结束，画个可视化相位图保存