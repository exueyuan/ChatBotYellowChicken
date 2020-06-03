/**
 * Created by 志俊 on 2016/5/30.
 */
$(function () {
    var $modal = $('#am-alert');
    $(".word_edit").click(function () {
        $id=$(this).attr("data-id");
        $('#am-edit-confirm').modal({
            onConfirm: function(options) {
                $("#word_edit_"+$id).hide();
                $("#word_sumbit_"+$id).show();
                $("#word_exit_"+$id).show();
                $("#word_delete_"+$id).hide();
                $("#category_item_text_"+$id).hide();
                $("#category_input_text_"+$id).show();
                $("#category_item_by_category_"+$id).hide();
                $("#category_input_by_category_"+$id).show();
            },
            onCancel: function() {
                alert('算求，不弄了');
            }
        });
    });
    $(".word_exit").click(function () {
        $id=$(this).attr("data-id");
        $('#am-exit-confirm').modal({
            onConfirm: function(options) {
                $("#word_edit_"+$id).show();
                $("#word_sumbit_"+$id).hide();
                $("#word_exit_"+$id).hide();
                $("#word_delete_"+$id).show();
                $("#category_item_text_"+$id).show();
                $("#category_input_text_"+$id).hide();
                $("#category_item_by_category_"+$id).show();
                $("#category_input_by_category_"+$id).hide();
            },
            onCancel: function() {
                alert('继续编辑');
            }
        });
    });
    $(".word_sumbit").click(function () {
        $id=$(this).attr("data-id");
        $text=$("#category_input_text_"+$id).val();
        $by_category=$("#category_input_by_category_"+$id).val();
        $.ajax({
            url:$app+"Admin/catergory_edit",
            type:"post",
            data:{id:$id,text:$text,by_catergory:$by_category},
            dataType:"json",
            success:function (data) {
                $modal.on('open.modal.amui', function(){
                    $("#am_alert_warning").html(data.info);
                });
                $modal.modal();
                $("#word_edit_"+$id).show();
                $("#word_sumbit_"+$id).hide();
                $("#word_exit_"+$id).hide();
                $("#word_delete_"+$id).show();
                $("#category_item_text_"+$id).html($("#category_input_text_"+$id).val());
                $("#category_item_text_"+$id).show();
                $("#category_input_text_"+$id).hide();
                $("#category_item_by_category_"+$id).html($("#category_input_by_category_"+$id).val());
                $("#category_item_by_category_"+$id).show();
                $("#category_input_by_category_"+$id).hide();
            }
        });
    });
    $(".word_delete").click(function () {
        $id=$(this).attr("data-id");
        $('#am-delete-confirm').modal({
            onConfirm: function(options) {
                $.ajax({
                    url:$app+"Admin/catergory_delete",
                    type:"post",
                    data:{id:$id},
                    dataType:"json",
                    success:function (data) {
                        if (data.status == 1) {
                            catergory_info="#category_info_"+$id;
                            $(catergory_info).remove();
                        }
                        else {
                            var $modal = $('#am-alert');
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
    $("#model_category_sumbit").click(function () {
        var $text=$('#model_category_text').val();
        var $by_catergory=$("#model_category_by_category").val();
        $.ajax({
            url:$app+"Admin/catergory_add",
            type:"post",
            data:{text:$text,by_catergory:$by_catergory},
            dataType:"json",
            success:function (data) {
                if (data.status == 1) {
                    var $modal = $('#add_word_model');
                    $modal.modal('close');
                    window.location.href=data.url;
                }
                else {
                    $("#model_warning").html(data.info);
                }
            }
        });
    })
});

