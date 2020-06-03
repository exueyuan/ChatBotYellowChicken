/**
 * Created by 志俊 on 2016/5/31.
 */
$(function () {
   $("#author_edit_btn").click(function () {
       $id=$("#author_id").val();
       $username=$("#author_username").val();
       $nickname=$("#author_nickname").val();
       $description=$("#author_description").val();
       $skill=$("#author_skill").val();
       $email=$("#author_email").val();
       $phone=$("#author_phone").val();
       $qq=$("#author_qq").val();
       $weixin=$("#author_weixin").val();
       $field=$("#author_field").val();
       $info=$("#author_info").val();
       $city=$("#author_city").val();
       $.ajax({
           url:$app+"Admin/author_edit_action",
           type:"post",
           data:
           {
               id:$id,
               username:$username,
               nickname:$nickname,
               description:$description,
               skill:$skill,
               email:$email,
               phone:$phone,
               weixin:$weixin,
               field:$field,
               info:$info,
               city:$city
           },
           dataType:"json",
           success:function (data) {
               var $modal = $('#am-alert');
               $modal.on('open.modal.amui', function(){
                   $("#am_alert_warning").html(data.info);
               });
               $modal.modal();
           }
       })
   });
    $(".author_delete").click(function () {
        $id=$(this).attr("data-id");
        $('#am-delete-confirm').modal({
            onConfirm: function(options) {
                $.ajax({
                    url:$app+"Admin/author_delete",
                    type:"post",
                    data:{id:$id},
                    dataType:"json",
                    success:function (data) {
                        if (data.status == 1) {
                            exarticle_info="#author_info_"+$id;
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
});