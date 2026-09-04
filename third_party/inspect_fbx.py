import sys
import fbx
import FbxCommon

def inspect_fbx(fbx_file):
    # 初始化FBX SDK
    fbx_sdk_manager, fbx_scene = FbxCommon.InitializeSdkObjects()
    FbxCommon.LoadScene(fbx_sdk_manager, fbx_scene, fbx_file)
    
    # 获取根节点
    root_node = fbx_scene.GetRootNode()
    
    print(f"检查FBX文件: {fbx_file}")
    print(f"场景根节点: {root_node.GetName()}")
    print(f"\n节点层级结构:")
    
    def print_hierarchy(node, level=0):
        indent = "  " * level
        node_name = node.GetName()
        node_type = node.GetTypeName()
        child_count = node.GetChildCount()
        
        # 检查是否有动画曲线
        has_anim = False
        try:
            num_anim_stacks = fbx_scene.GetSrcObjectCount(
                FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimStack.ClassId)
            )
            if num_anim_stacks > 0:
                anim_stack = fbx_scene.GetSrcObject(
                    FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimStack.ClassId), 0
                )
                num_anim_layers = anim_stack.GetSrcObjectCount(
                    FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimLayer.ClassId)
                )
                if num_anim_layers > 0:
                    animation_layer = anim_stack.GetSrcObject(
                        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimLayer.ClassId), 0
                    )
                    for c in ["X", "Y", "Z"]:
                        curve = node.LclTranslation.GetCurve(animation_layer, c)
                        if curve and curve.KeyGetCount() > 0:
                            has_anim = True
                            break
        except:
            pass
        
        anim_marker = " [有动画]" if has_anim else ""
        print(f"{indent}- {node_name} (类型: {node_type}, 子节点: {child_count}){anim_marker}")
        
        # 递归打印子节点
        for i in range(child_count):
            child = node.GetChild(i)
            print_hierarchy(child, level + 1)
    
    print_hierarchy(root_node)
    
    # 打印动画信息
    print(f"\n动画堆栈信息:")
    num_anim_stacks = fbx_scene.GetSrcObjectCount(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimStack.ClassId)
    )
    print(f"动画堆栈数量: {num_anim_stacks}")
    
    for i in range(num_anim_stacks):
        anim_stack = fbx_scene.GetSrcObject(
            FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimStack.ClassId), i
        )
        print(f"  堆栈 {i}: {anim_stack.GetName()}")
        
        anim_range = anim_stack.GetLocalTimeSpan()
        duration = anim_range.GetDuration()
        fps = duration.GetFrameRate(duration.GetGlobalTimeMode())
        frame_count = duration.GetFrameCount(True)
        print(f"    FPS: {fps}, 帧数: {frame_count}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python inspect_fbx.py <fbx文件路径>")
        sys.exit(1)
    
    inspect_fbx(sys.argv[1])
