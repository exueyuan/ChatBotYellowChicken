/**
 * Created by 志俊 on 2016/5/29 0029.
 */
$(function() {
    $(".spider_to_exarticle").click(function () {
        $id=$(this).attr('data-id');
        $source=$(this).attr('data-source');
        $('#am-add-confirm').modal({
            onConfirm: function(options) {
                $.ajax({
                    url:$app+"Admin/spider_to_article",
                    type:"post",
                    data:{id:$id,Source:$source},
                    dataType:"json",
                    success:function (data) {
                        var $modal = $('#am-alert');
                        $modal.on('open.modal.amui', function(){
                            $("#am_alert_warning").html(data.info);
                        });
                        $modal.modal();
                    }
                })
            },
            // closeOnConfirm: false,
            onCancel: function() {
                alert('算求，不弄了');
            }
        });
    });
});

