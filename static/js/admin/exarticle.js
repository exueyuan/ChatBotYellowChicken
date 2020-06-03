/**
 * Created by 志俊 on 2016/5/29 0029.
 */
$(function() {
    $(".exarticle_delete").click(function () {
        $id=$(this).attr("data-id");
        $('#am-delete-confirm').modal({
            onConfirm: function(options) {
                $.ajax({
                    url:$app+"Admin/exarticle_delete",
                    type:"post",
                    data:{id:$id},
                    dataType:"json",
                    success:function (data) {
                        if (data.status == 1) {
                            exarticle_info="#exarticle_info_"+$id;
                            $(exarticle_info).remove();
                        }
                        else {
                            var $modal = $('#exarticle-alert');
                            $modal.modal();
                            $modal.on('open.modal.amui', function(){
                                $("#am_alert_warning").html(data.info)
                            });
                        }
                    }
                })
            },
            // closeOnConfirm: false,
            onCancel: function() {
                alert('算求，不弄了');
            }
        });
    });
    $("#exarticel_edit_btn").click(function () {
        $id=$("#exarticle_id").val(); 
        $category=$("#exarticle_category").val();
        $text=$("#exarticle_text").val();
        $title=$("#exarticle_title").val();
        $.ajax({
            url:$app+"Admin/exarticle_edit_action",
            type:"post",
            data:{id:$id,title:$title,text:$text,catergory:$category},
            dataType:"json",
            success:function (data) {
                var $modal = $('#am-alert');
                $modal.on('open.modal.amui', function(){
                    $("#am_alert_warning").html(data.info);
                });
                $modal.modal();
            }
        });
    });
});
