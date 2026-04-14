# Nemo Agent ProGuard Rules
-keep class com.nemo.knowledge.** { *; }
-keep class com.nemo.screen.** { *; }
-keep class com.nemo.model.** { *; }
-keep class com.nemo.research.** { *; }
-keep class com.nemo.safety.** { *; }

# MediaPipe
-keep class com.google.mediapipe.** { *; }

# Room
-keep class * extends androidx.room.RoomDatabase
-keep @androidx.room.Entity class *
-keep @androidx.room.Dao class *
